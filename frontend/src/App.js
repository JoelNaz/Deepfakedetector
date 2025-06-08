// src/App.js
import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import DeepfakeDetector from './components/DeepfakeDetector';
import AnalysisHistory from './components/AnalysisHistory';

// Main App Logic Component
const AppContent = () => {
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'signup'
  const [currentView, setCurrentView] = useState('analysis'); // 'analysis' or 'history'
  const { user, loading } = useAuth();

  // Navigation functions
  const navigateToHistory = () => setCurrentView('history');
  const navigateToAnalysis = () => setCurrentView('analysis');

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 text-purple-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-300">Loading...</p>
        </div>
      </div>
    );
  }

  // Show auth pages if user is not logged in
  if (!user) {
    return authMode === 'login' ? (
      <LoginPage onSwitchToSignup={() => setAuthMode('signup')} />
    ) : (
      <SignupPage onSwitchToLogin={() => setAuthMode('login')} />
    );
  }

  // Show main application based on current view
  return currentView === 'analysis' ? (
    <DeepfakeDetector onNavigateToHistory={navigateToHistory} />
  ) : (
    <AnalysisHistory onNavigateToAnalysis={navigateToAnalysis} />
  );
};

// Root App Component with Auth Provider
const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;