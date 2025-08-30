from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import json
import io
import os
import numpy as np
from typing import List, Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom model classes (needed for loading the saved model)
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        attention = tf.matmul(weights, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'rate': self.rate})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({'maxlen': self.maxlen, 'vocab_size': self.vocab_size, 'embed_dim': self.embed_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class StyleMappingNetwork(keras.Model):
    def __init__(self, style_dim, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.style_dim = style_dim
        self.n_layers = n_layers
        model_layers = []
        for _ in range(n_layers):
            model_layers.append(layers.Dense(style_dim))
            model_layers.append(layers.BatchNormalization())
            model_layers.append(layers.LeakyReLU(alpha=0.2))
        self.mapping = keras.Sequential(model_layers)
    
    def call(self, x):
        return self.mapping(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({'style_dim': self.style_dim, 'n_layers': self.n_layers})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-8)
    return scale * vectors

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
    
    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[1, self.input_num_capsules, self.num_capsules, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )
        self.style_modulator = layers.Dense(self.dim_capsule * 2, name="style_modulator")
    
    def call(self, inputs, style_vector_w):
        u_hat = tf.squeeze(tf.matmul(self.W, tf.tile(tf.expand_dims(tf.expand_dims(inputs, 2), -1), [1, 1, self.num_capsules, 1, 1])), axis=-1)
        b = tf.zeros(shape=[tf.shape(u_hat)[0], self.input_num_capsules, self.num_capsules, 1])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            v = squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)
        style_params = self.style_modulator(style_vector_w)
        style_scale, style_bias = tf.split(style_params, 2, axis=-1)
        v_final = tf.squeeze(v, axis=1)
        v_modulated = v_final * (tf.expand_dims(style_scale, 1) + 1) + tf.expand_dims(style_bias, 1)
        return v_modulated
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_capsules': self.num_capsules, 'dim_capsule': self.dim_capsule, 'routings': self.routings})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DAC_Net_MultiHead(keras.Model):
    def __init__(self, maxlen, vocab_size, embed_dim=128, num_heads=4, ff_dim=512, num_transformer_blocks=2, style_layers=6, num_hallucination_types=4, capsule_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.style_layers = style_layers
        self.num_hallucination_types = num_hallucination_types
        self.capsule_dim = capsule_dim
        
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)]
        self.style_mapper = StyleMappingNetwork(style_dim=embed_dim, n_layers=style_layers)
        self.primary_capsules_conv = layers.Conv1D(filters=256, kernel_size=9, strides=2, padding='valid', activation='relu')
        self.primary_dim = 8
        self.hallucination_capsules = CapsuleLayer(num_capsules=num_hallucination_types, dim_capsule=capsule_dim)
        self.flatten = layers.Flatten()
        self.binary_classification_head = layers.Dense(2, activation='softmax', name='binary_output')
        self.type_classification_head = layers.Dense(num_hallucination_types, activation='softmax', name='type_output')
        self.prob_regression_head = layers.Dense(1, activation='sigmoid', name='prob_output')
    
    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x = block(x)
        sequence_output = x
        pooled_output = x[:, 0]
        style_vector_w = self.style_mapper(pooled_output)
        primary_caps_output = self.primary_capsules_conv(sequence_output)
        num_primary_caps = primary_caps_output.shape[1] * (primary_caps_output.shape[2] // self.primary_dim)
        primary_caps_reshaped = layers.Reshape((num_primary_caps, self.primary_dim))(primary_caps_output)
        primary_caps_squashed = squash(primary_caps_reshaped)
        final_capsules = self.hallucination_capsules(primary_caps_squashed, style_vector_w)
        flat_capsules = self.flatten(final_capsules)
        binary_pred = self.binary_classification_head(flat_capsules)
        type_pred = self.type_classification_head(flat_capsules)
        prob_pred = self.prob_regression_head(flat_capsules)
        reasoning_output = final_capsules
        return {'binary_output': binary_pred, 'type_output': type_pred, 'reasoning_output': reasoning_output, 'prob_output': prob_pred}
    
    def get_config(self):
        config = {
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "style_layers": self.style_layers,
            "num_hallucination_types": self.num_hallucination_types,
            "capsule_dim": self.capsule_dim
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

app = FastAPI(
    title="LLM Hallucination Validator API",
    description="API for validating LLM-generated content against reference prompts using ML models",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model at startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "dac_net_multihead_model.keras")
model = None
vectorizer = None

# Model constants
MAX_LEN = 256
VOCAB_SIZE = 20000

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Initialize text vectorizer with same parameters as training
        vectorizer = tf.keras.utils.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_sequence_length=MAX_LEN,
            standardize="lower_and_strip_punctuation"
        )
        logger.info("Text vectorizer initialized")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
        logger.info("Continuing with fallback prediction method")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.info("Continuing with fallback prediction method")

# Pydantic models for request/response
class SingleValidationRequest(BaseModel):
    generated_text: str
    prompt: str

class SingleValidationResponse(BaseModel):
    is_hallucination: bool
    percentage: float
    reasoning: str
    hallucination_type: Optional[str] = None

class BatchResultItem(BaseModel):
    entry_id: int
    generated_text: str
    prompt: str
    is_hallucination: bool
    percentage: float
    hallucination_type: str

class BatchValidationResponse(BaseModel):
    total_percentage: float
    total_entries: int
    hallucination_types: List[Dict[str, Any]]
    detailed_results: List[BatchResultItem]

# Helper functions for text preprocessing
def preprocess_text(text: str) -> str:
    """Clean and preprocess text for analysis"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.lower()

def extract_features(generated_text: str, prompt: str) -> Dict[str, float]:
    """Extract features for hallucination detection"""
    try:
        # Text preprocessing
        gen_clean = preprocess_text(generated_text)
        prompt_clean = preprocess_text(prompt)
        
        # Feature 1: Semantic similarity using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([prompt_clean, gen_clean])
        semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Feature 2: Length ratio
        length_ratio = len(gen_clean) / max(len(prompt_clean), 1)
        
        # Feature 3: Word overlap ratio
        prompt_words = set(prompt_clean.split())
        gen_words = set(gen_clean.split())
        word_overlap = len(prompt_words.intersection(gen_words)) / max(len(prompt_words.union(gen_words)), 1)
        
        # Feature 4: Sentence count ratio
        prompt_sentences = len([s for s in prompt_clean.split('.') if s.strip()])
        gen_sentences = len([s for s in gen_clean.split('.') if s.strip()])
        sentence_ratio = gen_sentences / max(prompt_sentences, 1)
        
        # Feature 5: Repetition detection
        words = gen_clean.split()
        unique_words = len(set(words))
        repetition_ratio = unique_words / max(len(words), 1)
        
        return {
            'semantic_similarity': semantic_similarity,
            'length_ratio': min(length_ratio, 5.0),  # Cap at 5x
            'word_overlap': word_overlap,
            'sentence_ratio': min(sentence_ratio, 3.0),  # Cap at 3x
            'repetition_ratio': repetition_ratio
        }
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return {
            'semantic_similarity': 0.5,
            'length_ratio': 1.0,
            'word_overlap': 0.3,
            'sentence_ratio': 1.0,
            'repetition_ratio': 0.8
        }

def predict_hallucination(features: Dict[str, float], input_text: str = "") -> tuple[bool, float, str]:
    """Predict hallucination using the loaded model or fallback logic"""
    try:
        if model is not None and vectorizer is not None and input_text:
            # Prepare text for the model
            processed_text = preprocess_text(input_text)
            
            # For now, we'll use a simple approach since we don't have the trained vectorizer
            # In a production system, you'd save and load the trained vectorizer
            try:
                # Create a simple tokenization approach
                words = processed_text.split()
                # Simple word to number mapping (this should be the trained vectorizer in production)
                word_to_idx = {word: idx + 1 for idx, word in enumerate(set(words))}
                tokenized = [word_to_idx.get(word, 0) for word in words[:MAX_LEN]]
                
                # Pad or truncate to MAX_LEN
                if len(tokenized) < MAX_LEN:
                    tokenized.extend([0] * (MAX_LEN - len(tokenized)))
                else:
                    tokenized = tokenized[:MAX_LEN]
                
                # Convert to tensor
                input_tensor = tf.expand_dims(tf.constant(tokenized, dtype=tf.int32), 0)
                
                # Get prediction from the model
                prediction = model(input_tensor)
                
                # Extract predictions
                binary_pred = prediction['binary_output']
                prob_pred = prediction['prob_output']
                type_pred = prediction['type_output']
                
                # Process binary prediction
                hallucination_class = tf.argmax(binary_pred, axis=-1).numpy()[0]
                is_hallucination = bool(hallucination_class == 1)
                
                # Process probability prediction
                hallucination_probability = float(prob_pred.numpy()[0][0])
                percentage = hallucination_probability * 100
                
                # Process type prediction
                type_class = tf.argmax(type_pred, axis=-1).numpy()[0]
                type_names = ["No Hallucination", "Factual Error", "Contextual Misalignment", "Logical Contradiction"]
                hallucination_type = type_names[min(type_class, len(type_names)-1)]
                
                logger.info(f"Model prediction: is_hallucination={is_hallucination}, percentage={percentage:.2f}")
                
                return is_hallucination, percentage, hallucination_type
                
            except Exception as model_error:
                logger.warning(f"Error during model prediction: {model_error}")
                # Fall back to feature-based prediction
                return predict_with_features(features)
        else:
            # Fallback logic when model is not available
            return predict_with_features(features)
            
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        # Return conservative fallback
        return True, 75.0, "Analysis Error"

def predict_with_features(features: Dict[str, float]) -> tuple[bool, float, str]:
    """Fallback prediction using extracted features"""
    hallucination_probability = calculate_fallback_score(features)
    is_hallucination = hallucination_probability > 0.5
    percentage = hallucination_probability * 100
    hallucination_type = determine_hallucination_type(features, is_hallucination)
    return is_hallucination, percentage, hallucination_type

def calculate_fallback_score(features: Dict[str, float]) -> float:
    """Fallback scoring when ML model is not available"""
    # Weighted scoring based on features
    weights = {
        'semantic_similarity': -0.4,  # Higher similarity = lower hallucination
        'word_overlap': -0.3,        # Higher overlap = lower hallucination  
        'length_ratio': 0.1,         # Very long responses might be hallucination
        'sentence_ratio': 0.1,       # Too many sentences might indicate hallucination
        'repetition_ratio': -0.1     # Higher repetition = potential hallucination
    }
    
    score = 0.5  # Base score
    for feature, value in features.items():
        if feature in weights:
            score += weights[feature] * (value - 0.5)  # Normalize around 0.5
    
    return max(0.0, min(1.0, score))  # Clamp between 0 and 1

def determine_hallucination_type(features: Dict[str, float], is_hallucination: bool) -> str:
    """Determine the type of hallucination based on features"""
    if not is_hallucination:
        return "No Hallucination"
    
    if features['semantic_similarity'] < 0.3:
        return "Contextual Misalignment"
    elif features['word_overlap'] < 0.2:
        return "Factual Error"
    elif features['length_ratio'] > 2.0:
        return "Content Expansion"
    elif features['repetition_ratio'] < 0.6:
        return "Repetitive Content"
    else:
        return "Logical Contradiction"

def generate_reasoning(features: Dict[str, float], is_hallucination: bool, percentage: float) -> str:
    """Generate human-readable reasoning for the prediction"""
    if is_hallucination:
        reasons = []
        if features['semantic_similarity'] < 0.4:
            reasons.append("low semantic similarity to the original prompt")
        if features['word_overlap'] < 0.3:
            reasons.append("limited word overlap with the source context")
        if features['length_ratio'] > 2.0:
            reasons.append("excessive content expansion beyond the prompt scope")
        if features['repetition_ratio'] < 0.7:
            reasons.append("repetitive or redundant content patterns")
        
        if reasons:
            return f"Hallucination detected ({percentage:.1f}% confidence) due to: {', '.join(reasons)}. The generated text appears to deviate significantly from the given prompt context."
        else:
            return f"Hallucination detected ({percentage:.1f}% confidence). The generated text shows signs of factual inaccuracies or unsupported claims."
    else:
        return f"No significant hallucination detected ({100-percentage:.1f}% confidence). The generated text aligns well with the given prompt and maintains contextual consistency."

# API Endpoints
@app.get("/")
async def root():
    return {"message": "LLM Hallucination Validator API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/validate/single", response_model=SingleValidationResponse)
async def validate_single(request: SingleValidationRequest):
    """Validate a single generated text against its prompt"""
    try:
        if not request.generated_text.strip() or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Both generated_text and prompt are required")
        
        # Extract features
        features = extract_features(request.generated_text, request.prompt)
        
        # Combine texts for model input (prompt + generated text)
        combined_text = f"{request.prompt} [SEP] {request.generated_text}"
        
        # Predict hallucination
        is_hallucination, percentage, hallucination_type = predict_hallucination(features, combined_text)
        
        # Generate reasoning
        reasoning = generate_reasoning(features, is_hallucination, percentage)
        
        return SingleValidationResponse(
            is_hallucination=is_hallucination,
            percentage=round(percentage, 2),
            reasoning=reasoning,
            hallucination_type=hallucination_type
        )
    
    except Exception as e:
        logger.error(f"Error in single validation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/validate/batch", response_model=BatchValidationResponse)
async def validate_batch(file: UploadFile = File(...)):
    """Validate multiple entries from CSV or JSON file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        
        # Parse file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or JSON.")
        
        # Validate required columns
        required_columns = ['generated_text', 'prompt']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        
        # Process each entry
        results = []
        hallucination_counts = {}
        total_percentage = 0
        
        for idx, row in df.iterrows():
            try:
                generated_text = str(row['generated_text']).strip()
                prompt = str(row['prompt']).strip()
                
                if not generated_text or not prompt:
                    continue
                
                # Extract features and predict
                features = extract_features(generated_text, prompt)
                combined_text = f"{prompt} [SEP] {generated_text}"
                is_hallucination, percentage, hallucination_type = predict_hallucination(features, combined_text)
                
                results.append(BatchResultItem(
                    entry_id=idx + 1,
                    generated_text=generated_text,
                    prompt=prompt,
                    is_hallucination=is_hallucination,
                    percentage=round(percentage, 2),
                    hallucination_type=hallucination_type
                ))
                
                # Update statistics
                total_percentage += percentage
                if hallucination_type in hallucination_counts:
                    hallucination_counts[hallucination_type] += 1
                else:
                    hallucination_counts[hallucination_type] = 1
                    
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid entries found in the file")
        
        # Calculate summary statistics
        avg_percentage = total_percentage / len(results)
        total_entries = len(results)
        
        # Format hallucination types for response
        hallucination_types = []
        for h_type, count in hallucination_counts.items():
            percentage = (count / total_entries) * 100
            hallucination_types.append({
                "type": h_type,
                "percentage": round(percentage, 2),
                "count": count
            })
        
        return BatchValidationResponse(
            total_percentage=round(avg_percentage, 2),
            total_entries=total_entries,
            hallucination_types=hallucination_types,
            detailed_results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is not None:
        return {
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "model_summary": str(model.summary()) if hasattr(model, 'summary') else "N/A"
        }
    else:
        return {
            "model_loaded": False,
            "model_path": MODEL_PATH,
            "error": "Model not loaded"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)