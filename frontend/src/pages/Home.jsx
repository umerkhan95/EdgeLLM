import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Key, Shield, Zap, BarChart } from 'lucide-react';

const Home = () => {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Ollama API Gateway
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            Secure, scalable, and monitored access to Ollama AI models
          </p>

          {!user && (
            <Link
              to="/signin"
              className="inline-flex items-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-lg font-medium"
            >
              <Key className="h-5 w-5 mr-2" />
              Sign In with API Key
            </Link>
          )}
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary-100 dark:bg-primary-900 mb-4">
              <Shield className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Secure Authentication
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              API key-based authentication with role-based access control
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary-100 dark:bg-primary-900 mb-4">
              <Zap className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Rate Limiting
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Configurable rate limits to prevent abuse and manage resources
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary-100 dark:bg-primary-900 mb-4">
              <BarChart className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Usage Analytics
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Detailed statistics and monitoring for all API requests
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary-100 dark:bg-primary-900 mb-4">
              <Key className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              API Management
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Create, manage, and revoke API keys with ease
            </p>
          </div>
        </div>

        <div className="mt-16 bg-white dark:bg-gray-800 rounded-lg shadow-md p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Quick Start
          </h2>
          <div className="space-y-4 text-gray-600 dark:text-gray-300">
            <div className="flex items-start">
              <span className="flex items-center justify-center h-6 w-6 rounded-full bg-primary-600 text-white text-sm font-medium mr-3">
                1
              </span>
              <p>Sign in with your API key or request one from an administrator</p>
            </div>
            <div className="flex items-start">
              <span className="flex items-center justify-center h-6 w-6 rounded-full bg-primary-600 text-white text-sm font-medium mr-3">
                2
              </span>
              <p>Access your dashboard to view usage statistics and manage your account</p>
            </div>
            <div className="flex items-start">
              <span className="flex items-center justify-center h-6 w-6 rounded-full bg-primary-600 text-white text-sm font-medium mr-3">
                3
              </span>
              <p>Use your API key to make requests to Ollama models through our gateway</p>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-gray-100 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            API Endpoint
          </h3>
          <code className="block bg-gray-800 dark:bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto">
            http://localhost:8000/api/
          </code>
          <p className="mt-3 text-sm text-gray-600 dark:text-gray-400">
            Include your API key in the <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">X-API-Key</code> header
          </p>
        </div>
      </div>
    </div>
  );
};

export default Home;
