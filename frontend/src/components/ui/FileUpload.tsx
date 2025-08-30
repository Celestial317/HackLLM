import React, { useCallback } from 'react';
import { Upload, File, X } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  acceptedTypes: string;
  description: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  selectedFile,
  acceptedTypes,
  description,
}) => {
  const [isDragOver, setIsDragOver] = React.useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const removeFile = () => {
    onFileSelect(null);
  };

  if (selectedFile) {
    return (
      <div className="border-2 border-primary-500 border-dashed rounded-xl p-8 text-center bg-primary-500/5">
        <div className="flex items-center justify-center space-x-4">
          <File className="w-8 h-8 text-primary-400" />
          <div className="flex-1 text-left">
            <p className="text-white font-medium">{selectedFile.name}</p>
            <p className="text-gray-400 text-sm">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </p>
          </div>
          <button
            onClick={removeFile}
            className="p-2 text-gray-400 hover:text-white hover:bg-dark-700 rounded-lg transition-colors duration-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 cursor-pointer ${
        isDragOver
          ? 'border-primary-400 bg-primary-500/10'
          : 'border-dark-600 hover:border-primary-500/50 bg-dark-800/50'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragOver ? 'text-primary-400' : 'text-gray-400'}`} />
      <h3 className="text-xl font-semibold text-white mb-2">
        {isDragOver ? 'Drop your file here' : 'Upload Dataset'}
      </h3>
      <p className="text-gray-400 mb-4">
        Drag and drop your file here, or click to browse
      </p>
      <p className="text-sm text-gray-500">{description}</p>
      
      <input
        id="file-input"
        type="file"
        accept={acceptedTypes}
        onChange={handleFileInput}
        className="hidden"
      />
    </div>
  );
};

export default FileUpload;