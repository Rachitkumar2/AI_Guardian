import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Library from './pages/Library';
import Tools from './pages/Tools';
import DashboardLayout from './components/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Signup from './pages/Signup';

function App() {
  return (
    <div className="min-h-screen bg-[#0E1511] text-white font-sans selection:bg-[#00FF66] selection:text-black flex flex-col">
      <Routes>
        {/* Isolated full-screen pages */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* Main Website Layout */}
        <Route path="/" element={
          <div className="flex flex-col min-h-screen">
            <Navbar />
            <main className="flex-grow">
              <Home />
            </main>
            <Footer />
          </div>
        } />
        
        <Route path="/library" element={
          <div className="flex flex-col min-h-screen">
            <Navbar active="library" />
            <main className="flex-grow">
              <Library />
            </main>
            <Footer />
          </div>
        } />

        <Route path="/tools" element={
          <div className="flex flex-col min-h-screen">
            <Navbar active="tools" />
            <main className="flex-grow">
              <Tools />
            </main>
            <Footer />
          </div>
        } />
        
        {/* App Dashboard Layout */}
        <Route path="/app" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          {/* Add more nested routes here if needed */}
        </Route>
      </Routes>
    </div>
  );
}

export default App;
