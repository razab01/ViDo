import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './features/auth';
import { LandingPage } from './features/landing/LandingPage';
import { StyleLab } from './features/style-lab/StyleLab';
import { WorkspaceLayout } from './features/workspace/WorkspaceLayout';
import { NotFoundView } from './features/workspace/views/NotFoundView';
import { ErrorBoundary } from './components/ui';
import { ThemeProvider } from './features/theme/ThemeProvider';
import { SeasonalBackdrop } from './features/theme/SeasonalBackdrop';

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <div className="vdb-app-shell">
          <SeasonalBackdrop />
          <AuthProvider>
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/style-lab" element={<StyleLab />} />
                <Route path="/ml-lab" element={<Navigate to="/workspace/ml-lab" replace />} />
                <Route path="/converse" element={<Navigate to="/workspace/converse" replace />} />
                <Route path="/chat-zone" element={<Navigate to="/workspace/converse" replace />} />
                <Route path="/chat-lab" element={<Navigate to="/workspace/converse" replace />} />
                <Route path="/preview/*" element={<WorkspaceLayout isPreview={true} />} />
                <Route path="/workspace/*" element={<WorkspaceLayout isPreview={false} />} />
                <Route path="/dashboard" element={<Navigate to="/workspace" replace />} />
                <Route path="*" element={<NotFoundView />} />
              </Routes>
            </BrowserRouter>
          </AuthProvider>
        </div>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
