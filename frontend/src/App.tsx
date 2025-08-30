import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import SingleValidator from './pages/SingleValidator';
import BatchValidator from './pages/BatchValidator';
import About from './pages/About';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-dark text-white">
        <Navbar />
        <main className="pt-20">
          <Routes>
            <Route path="/" element={<SingleValidator />} />
            <Route path="/batch" element={<BatchValidator />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;