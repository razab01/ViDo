import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import {
  applyThemeAttributes,
  buildSceneFromKey,
  getSeasonalScene,
  readSceneOverride,
  readThemeChoice,
  SCENE_STORAGE_KEY,
  THEME_STORAGE_KEY,
  writeSceneOverride,
  writeThemeChoice,
  type SceneOverride,
  type SeasonalScene,
  type ThemeChoice,
} from './theme';

interface ThemeContextValue {
  theme: ThemeChoice;
  scene: SeasonalScene;
  autoScene: SeasonalScene;
  sceneOverride: SceneOverride;
  setTheme: (choice: ThemeChoice) => void;
  setSceneOverride: (choice: SceneOverride) => void;
  refreshScene: () => void;
}

const fallbackAutoScene = getSeasonalScene();
const fallbackScene = { ...fallbackAutoScene };

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'general',
  scene: fallbackScene,
  autoScene: fallbackAutoScene,
  sceneOverride: 'auto',
  setTheme: () => {},
  setSceneOverride: () => {},
  refreshScene: () => {},
});

export const useTheme = (): ThemeContextValue => useContext(ThemeContext);

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeChoice>(() => readThemeChoice());
  const [autoScene, setAutoScene] = useState<SeasonalScene>(() => getSeasonalScene());
  const [sceneOverride, setSceneOverrideState] = useState<SceneOverride>(() => readSceneOverride());

  const scene = useMemo(
    () => (sceneOverride === 'auto' ? autoScene : buildSceneFromKey(sceneOverride, autoScene)),
    [autoScene, sceneOverride],
  );

  const refreshScene = useCallback(() => {
    setAutoScene(getSeasonalScene());
  }, []);

  const setTheme = useCallback((choice: ThemeChoice) => {
    setThemeState(choice);
    writeThemeChoice(choice);
  }, []);

  const setSceneOverride = useCallback((choice: SceneOverride) => {
    setSceneOverrideState(choice);
    writeSceneOverride(choice);
  }, []);

  useEffect(() => {
    applyThemeAttributes(theme, scene);
  }, [theme, scene]);

  useEffect(() => {
    if (theme !== 'seasonal') return;
    const timer = window.setInterval(refreshScene, 5 * 60 * 1000);
    return () => window.clearInterval(timer);
  }, [theme, refreshScene]);

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === THEME_STORAGE_KEY) {
        setThemeState(readThemeChoice());
      }
      if (event.key === SCENE_STORAGE_KEY) {
        setSceneOverrideState(readSceneOverride());
      }
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const value = useMemo(
    () => ({
      theme,
      scene,
      autoScene,
      sceneOverride,
      setTheme,
      setSceneOverride,
      refreshScene,
    }),
    [theme, scene, autoScene, sceneOverride, setTheme, setSceneOverride, refreshScene],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};
