// src/components/DeepfakeDetector.js
import React, { useState } from 'react';
import { Upload, Play, AlertCircle, CheckCircle, X, Loader2, LogOut, History } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const DeepfakeDetector = ({ onNavigateToHistory }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const { user, logout } = useAuth();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        setError('');
      } else {
        setError('Please select a valid video file');
      }
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        setError('');
      } else {
        setError('Please select a valid video file');
      }
    }
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/analyze-video/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to analyze video');
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setResult(data);
      } else {
        throw new Error(data.message || 'Analysis failed');
      }
    } catch (err) {
      setError('Error analyzing video: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const resetAll = () => {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setIsLoading(false);
  };

  const formatConfidence = (confidence) => {
    return (confidence * 100).toFixed(1);
  };

  const getResultColor = (isFake) => {
    return isFake ? 'text-red-600' : 'text-green-600';
  };

  const getResultBgColor = (isFake) => {
    return isFake ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header with user info and navigation */}
      <div className="bg-white/5 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white">AI Deepfake Detector</h1>
          <div className="flex items-center space-x-4">
            <span className="text-slate-300">
              Welcome, {user?.first_name} {user?.last_name}
            </span>
            <button
              onClick={onNavigateToHistory}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200"
            >
              <History className="h-4 w-4" />
              <span>History</span>
            </button>
            <button
              onClick={logout}
              className="flex items-center space-x-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors duration-200"
            >
              <LogOut className="h-4 w-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Welcome message */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-white mb-4">
            Detect Deepfake Content
          </h2>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto">
            Upload a video file to analyze whether it contains deepfake content using advanced AI detection algorithms
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {!result ? (
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20">
              {/* File Upload Area */}
              <div
                className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
                  dragActive
                    ? 'border-purple-400 bg-purple-500/10'
                    : 'border-slate-400 hover:border-purple-400 hover:bg-purple-500/5'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  disabled={isLoading}
                />
                
                {!selectedFile ? (
                  <div className="space-y-4">
                    <Upload className="mx-auto h-16 w-16 text-slate-400" />
                    <div>
                      <p className="text-xl font-semibold text-white mb-2">
                        Drop your video here or click to browse
                      </p>
                      <p className="text-slate-400">
                        Supports MP4, AVI, MOV and other video formats
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Play className="mx-auto h-16 w-16 text-purple-400" />
                    <div>
                      <p className="text-xl font-semibold text-white mb-2">
                        {selectedFile.name}
                      </p>
                      <p className="text-slate-400">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Error Message */}
              {error && (
                <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center space-x-3">
                  <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
                  <p className="text-red-400">{error}</p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-center space-x-4 mt-8">
                {selectedFile && (
                  <button
                    onClick={resetAll}
                    className="px-6 py-3 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors duration-200 flex items-center space-x-2"
                    disabled={isLoading}
                  >
                    <X className="h-5 w-5" />
                    <span>Clear</span>
                  </button>
                )}
                
                <button
                  onClick={uploadVideo}
                  disabled={!selectedFile || isLoading}
                  className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all duration-200 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5" />
                      <span>Analyze Video</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            /* Results Display */
            <div className="space-y-6">
              {/* Main Result Card */}
              <div className={`rounded-2xl p-8 border-2 ${getResultBgColor(result.is_fake)}`}>
                <div className="text-center">
                  <div className="flex justify-center mb-4">
                    {result.is_fake ? (
                      <AlertCircle className="h-16 w-16 text-red-600" />
                    ) : (
                      <CheckCircle className="h-16 w-16 text-green-600" />
                    )}
                  </div>
                  
                  <h2 className={`text-4xl font-bold mb-2 ${getResultColor(result.is_fake)}`}>
                    {result.is_fake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC VIDEO'}
                  </h2>
                  
                  <p className="text-xl text-slate-700 mb-6">
                    {result.is_fake 
                      ? 'This video appears to contain synthetic or manipulated content'
                      : 'This video appears to be authentic and unmodified'
                    }
                  </p>

                  <div className="bg-white/50 rounded-lg p-4 inline-block">
                    <p className="text-sm text-slate-600 mb-1">Confidence Score</p>
                    <p className={`text-3xl font-bold ${getResultColor(result.is_fake)}`}>
                      {formatConfidence(result.confidence)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Technical Details */}
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <h3 className="text-xl font-semibold text-white mb-4">Analysis Details</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-slate-300">
                  <div>
                    <p className="text-sm text-slate-400">Filename</p>
                    <p className="font-medium">{result.filename}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400">Raw Score</p>
                    <p className="font-medium">{result.raw_score?.toFixed(4) || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400">Threshold</p>
                    <p className="font-medium">0.5</p>
                  </div>
                </div>
              </div>

              {/* New Analysis Button */}
              <div className="text-center">
                <button
                  onClick={resetAll}
                  className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all duration-200 flex items-center space-x-2 mx-auto"
                >
                  <Upload className="h-5 w-5" />
                  <span>Analyze Another Video</span>
                </button>
              </div>
            </div>
          )}

          {/* Loading Overlay */}
          {isLoading && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
              <div className="bg-white rounded-2xl p-8 text-center max-w-md mx-4">
                <Loader2 className="h-16 w-16 text-purple-600 animate-spin mx-auto mb-4" />
                <h3 className="text-2xl font-semibold text-slate-800 mb-2">
                  Analyzing Video...
                </h3>
                <p className="text-slate-600">
                  Our AI is processing your video to detect deepfake content. This may take a few moments.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DeepfakeDetector;