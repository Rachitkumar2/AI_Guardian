import { Navigate, Outlet } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const user = localStorage.getItem('user');
  
  if (!user) {
    // Redirect to login if user is not found in localStorage
    return <Navigate to="/login" replace />;
  }

  // Render children if provided, otherwise render the Outlet for nested routes
  return children ? children : <Outlet />;
};

export default ProtectedRoute;
