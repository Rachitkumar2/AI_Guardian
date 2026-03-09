import { useState, useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import { User, Shield, Key } from 'lucide-react';

export default function SettingsLayout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="grow max-w-7xl mx-auto w-full px-4 sm:px-8 py-10 flex flex-col md:flex-row gap-8">
        
        {/* Sidebar */}
        <aside className="w-full md:w-64 shrink-0">
          <div className="mb-8">
            <h1 className="text-2xl font-bold mb-1">Settings</h1>
            <p className="text-sm text-gray-500">Manage your security environment</p>
          </div>
          
          <nav className="flex flex-col gap-2">
            <NavLink 
              to="/settings/profile"
              className={({ isActive }) => 
                `flex items-center gap-3 px-4 py-2.5 rounded-lg font-medium transition-colors ${
                  isActive 
                    ? 'bg-[#153A24] text-neon-green' 
                    : 'text-[#8B9A92] hover:text-white'
                }`
              }
            >
              <User className="w-5 h-5" />
              Profile
            </NavLink>
            
            <NavLink 
              to="/settings/security"
              className={({ isActive }) => 
                `flex items-center gap-3 px-4 py-2.5 rounded-lg font-medium transition-colors ${
                  isActive 
                    ? 'bg-[#153A24] text-neon-green' 
                    : 'text-[#8B9A92] hover:text-white'
                }`
              }
            >
              <Shield className="w-5 h-5" />
              Security
            </NavLink>
          </nav>
        </aside>

        {/* Content Area */}
        <div className="flex-1 w-full">
          <Outlet />
        </div>
        
      </main>

      <Footer />
    </div>
  );
}
