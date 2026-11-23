import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './features/auth';
import { LandingPage } from './features/landing/LandingPage';
import { StyleLab } from './features/style-lab/StyleLab';
import { WorkspaceLayout } from './features/workspace/WorkspaceLayout';
import { NotFoundView } from './features/workspace/views/NotFoundView';
import { ErrorBoundary } from './components/ui';
import { scheduleLocalModelWarmup } from './features/ml-lab/languageModel';
import { hasAnySavedSnapshot } from './features/ml-lab/mlStore';

export default function App() {
  useEffect(() => {
    // Download the answer assistant into the browser cache once, in the
    // background, whenever this account has saved data to ask about.
    if (hasAnySavedSnapshot()) scheduleLocalModelWarmup();
  }, []);

  return (
    <ErrorBoundary>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/style-lab" element={<StyleLab />} />
            <Route path="/ml-lab" element={<Navigate to="/workspace/ml-lab" replace />} />
            <Route path="/preview/*" element={<WorkspaceLayout isPreview={true} />} />
            <Route path="/workspace/*" element={<WorkspaceLayout isPreview={false} />} />
            <Route path="/dashboard" element={<Navigate to="/workspace" replace />} />
            <Route path="*" element={<NotFoundView />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ErrorBoundary>
  );
}
