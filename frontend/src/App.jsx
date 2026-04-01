import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Library from './pages/Library';
import Tools from './pages/Tools';
import DashboardLayout from './components/DashboardLayout';
import Dashboard from './pages/Dashboard';
import DetectionHistory from './pages/DetectionHistory';
import Login from './pages/Login';
import Signup from './pages/Signup';
import SettingsLayout from './components/SettingsLayout';
import ProfileSettings from './pages/settings/Profile';
import SecuritySettings from './pages/settings/Security';

function App() {
  return (
    <div className="min-h-screen bg-dark-bg text-white font-sans selection:bg-neon-green selection:text-black flex flex-col">
      <Routes>
        {/* Isolated full-screen pages */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* Main Website Layout */}
        <Route path="/" element={
          <div className="flex flex-col min-h-screen">
            <Navbar />
            <main className="grow">
              <Home />
            </main>
            <Footer />
          </div>
        } />

        <Route path="/library" element={
          <div className="flex flex-col min-h-screen">
            <Navbar active="library" />
            <main className="grow">
              <Library />
            </main>
            <Footer />
          </div>
        } />

        <Route path="/tools" element={
          <div className="flex flex-col min-h-screen">
            <Navbar active="tools" />
            <main className="grow">
              <Tools />
            </main>
            <Footer />
          </div>
        } />

        {/* App Dashboard Layout */}
        <Route path="/app" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="history" element={<DetectionHistory />} />
          {/* Add more nested routes here if needed */}
        </Route>

        {/* Settings Layout */}
        <Route path="/settings" element={<SettingsLayout />}>
          <Route path="profile" element={<ProfileSettings />} />
          <Route path="security" element={<SecuritySettings />} />
        </Route>
      </Routes>
    </div>
  );
}

export default App;
