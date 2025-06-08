// src/components/AnalysisHistory.js
import React, { useState, useEffect } from 'react';
import { Calendar, Clock, CheckCircle, XCircle, Filter, Search, Upload, LogOut, ArrowLeft } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const AnalysisHistory = ({ onNavigateToAnalysis }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({});
  const [pagination, setPagination] = useState({});
  const [filters, setFilters] = useState({
    result: 'all',
    days: 'all',
    search: ''
  });
  const [currentPage, setCurrentPage] = useState(1);
  const { user, logout } = useAuth();

  const fetchHistory = async (page = 1) => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: '10'
      });
      
      if (filters.result !== 'all') params.append('result', filters.result);
      if (filters.days !== 'all') params.append('days', filters.days);
      
      const response = await fetch(`http://localhost:8000/api/history/?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (data.success) {
        setHistory(data.history);
        setStats(data.stats);
        setPagination(data.pagination);
        setCurrentPage(page);
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory(1);
  }, [filters]);

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({ ...prev, [filterType]: value }));
  };

  const filteredHistory = history.filter(item =>
    item.filename.toLowerCase().includes(filters.search.toLowerCase())
  );

  const getResultBadge = (isDeepfake, confidence) => {
    if (isDeepfake) {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
          <XCircle className="w-3 h-3 mr-1" />
          Deepfake ({confidence}%)
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          <CheckCircle className="w-3 h-3 mr-1" />
          Real ({confidence}%)
        </span>
      );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-white/5 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <button
              onClick={onNavigateToAnalysis}
              className="flex items-center space-x-2 text-slate-300 hover:text-white transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
              <span>Back to Analysis</span>
            </button>
            <h1 className="text-2xl font-bold text-white">Analysis History</h1>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-slate-300">
              Welcome, {user?.first_name} {user?.last_name}
            </span>
            <button
              onClick={onNavigateToAnalysis}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200"
            >
              <Upload className="h-4 w-4" />
              <span>New Analysis</span>
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
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-4">Your Detection History</h2>
          <p className="text-xl text-slate-300">View and manage your past deepfake detection results</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 p-4">
            <div className="text-2xl font-bold text-white">{stats.total_analyses || 0}</div>
            <div className="text-sm text-slate-300">Total Analyses</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 p-4">
            <div className="text-2xl font-bold text-green-400">{stats.real_count || 0}</div>
            <div className="text-sm text-slate-300">Real Videos</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 p-4">
            <div className="text-2xl font-bold text-red-400">{stats.fake_count || 0}</div>
            <div className="text-sm text-slate-300">Deepfakes</div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 p-4 mb-6">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-300" />
              <span className="font-medium text-white">Filters:</span>
            </div>
            
            <select
              value={filters.result}
              onChange={(e) => handleFilterChange('result', e.target.value)}
              className="bg-white/5 border border-white/20 rounded px-3 py-1 text-sm text-white"
            >
              <option value="all" className="bg-gray-900 text-white">All Results</option>
              <option value="real" className="bg-gray-900 text-white">Real Only</option>
              <option value="fake" className="bg-gray-900 text-white">Deepfake Only</option>
            </select>
            
            <select
              value={filters.days}
              onChange={(e) => handleFilterChange('days', e.target.value)}
              className="bg-white/5 border border-white/20 rounded px-3 py-1 text-sm text-white"
            >
              <option value="all" className="bg-gray-900 text-white">All Time</option>
              <option value="7" className="bg-gray-900 text-white">Last 7 Days</option>
              <option value="30" className="bg-gray-900 text-white">Last 30 Days</option>
              <option value="90" className="bg-gray-900 text-white">Last 90 Days</option>
            </select>
            
            <div className="flex items-center gap-2">
              <Search className="w-4 h-4 text-slate-300" />
              <input
                type="text"
                placeholder="Search videos..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="bg-white/5 border border-white/20 rounded px-3 py-1 text-sm w-48 text-white placeholder-slate-400"
              />
            </div>
          </div>
        </div>

        {/* History Table */}
        <div className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 overflow-hidden">
          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto"></div>
              <p className="mt-2 text-slate-300">Loading history...</p>
            </div>
          ) : filteredHistory.length === 0 ? (
            <div className="p-8 text-center text-slate-300">
              <Upload className="h-12 w-12 mx-auto mb-4 text-slate-400" />
              <p className="text-lg mb-2">No analysis history found</p>
              <p className="text-sm text-slate-400">Upload and analyze some videos to see them here!</p>
              <button
                onClick={onNavigateToAnalysis}
                className="mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                Start Analyzing
              </button>
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-white/10">
                  <thead className="bg-white/5">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                        Video Name
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                        Date & Time
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                        Result
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                        Details
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10">
                    {filteredHistory.map((item) => (
                      <tr key={item.id} className="hover:bg-white/5 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-white">
                            {item.filename}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                          <div className="flex items-center gap-1">
                            <Calendar className="w-4 h-4" />
                            {item.date_only}
                          </div>
                          <div className="flex items-center gap-1 mt-1">
                            <Clock className="w-4 h-4" />
                            {item.time_only}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {getResultBadge(item.is_deepfake, item.confidence)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                          <div>Score: {item.prediction_score}</div>
                          <div>Duration: {item.analysis_duration}s</div>
                          <div>Frames: {item.frames_analyzed}</div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {pagination.total_pages > 1 && (
                <div className="bg-white/5 px-4 py-3 border-t border-white/10 sm:px-6">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-300">
                      Showing page {pagination.page} of {pagination.total_pages} 
                      ({pagination.total_count} total results)
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => fetchHistory(currentPage - 1)}
                        disabled={!pagination.has_previous}
                        className="px-3 py-1 bg-white/10 border border-white/20 rounded text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/20 transition-colors"
                      >
                        Previous
                      </button>
                      <button
                        onClick={() => fetchHistory(currentPage + 1)}
                        disabled={!pagination.has_next}
                        className="px-3 py-1 bg-white/10 border border-white/20 rounded text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/20 transition-colors"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisHistory;