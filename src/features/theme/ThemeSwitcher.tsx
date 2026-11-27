import React, { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, CloudRain, Moon, Palette, Snowflake, Sun } from 'lucide-react';
import { useTheme } from './ThemeProvider';
import { SceneArtwork } from './SeasonalBackdrop';
import { buildSceneFromKey, SCENE_KEYS, SCENE_LABELS, type SceneKey } from './theme';

interface ThemeSwitcherProps {
  compact?: boolean;
}

const SceneIcon: React.FC<{ scene: string; className?: string }> = ({ scene, className }) => {
  if (scene.startsWith('winter')) return <Snowflake className={className} />;
  if (scene.startsWith('rainy')) return <CloudRain className={className} />;
  if (scene.endsWith('night')) return <Moon className={className} />;
  return <Sun className={className} />;
};

export const ThemeSwitcher: React.FC<ThemeSwitcherProps> = ({ compact = false }) => {
  const { theme, scene, autoScene, sceneOverride, setTheme, setSceneOverride } = useTheme();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handlePointer = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', handlePointer);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handlePointer);
      document.removeEventListener('keydown', handleKey);
    };
  }, [open]);

  const generalActive = theme === 'general';
  const autoActive = theme === 'seasonal' && sceneOverride === 'auto';
  const triggerLabel =
    theme === 'general' ? 'General' : scene.isPreview ? `${scene.label} preview` : scene.label;

  const selectScene = (key: SceneKey) => {
    setTheme('seasonal');
    setSceneOverride(key);
    setOpen(false);
  };

  return (
    <div ref={rootRef} className="vdb-theme-switcher">
      <button
        type="button"
        className={`vdb-theme-trigger${compact ? ' vdb-theme-trigger--compact' : ''}`}
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={theme === 'general' ? 'Theme: General' : `Theme: ${triggerLabel}`}
      >
        {theme === 'general' ? (
          <Palette className="vdb-theme-trigger-icon" />
        ) : (
          <SceneIcon scene={scene.key} className="vdb-theme-trigger-icon" />
        )}
        {!compact && <span className="vdb-theme-trigger-label">{triggerLabel}</span>}
        <ChevronDown className="vdb-theme-chevron" />
      </button>

      {open && (
        <div className="vdb-theme-menu" role="listbox" aria-label="Theme">
          <p className="vdb-theme-menu-heading">Themes</p>

          <button
            type="button"
            role="option"
            aria-selected={generalActive}
            className={`vdb-theme-option${generalActive ? ' is-active' : ''}`}
            onClick={() => {
              setTheme('general');
              setOpen(false);
            }}
          >
            <span className="vdb-theme-option-icon vdb-theme-option-icon--general">
              <Palette />
            </span>
            <span className="vdb-theme-option-copy">
              <strong>General</strong>
              <small>The warm editorial theme.</small>
            </span>
            {generalActive && <Check className="vdb-theme-check" />}
          </button>

          <button
            type="button"
            role="option"
            aria-selected={autoActive}
            className={`vdb-theme-option${autoActive ? ' is-active' : ''}`}
            onClick={() => {
              setTheme('seasonal');
              setSceneOverride('auto');
              setOpen(false);
            }}
          >
            <span className="vdb-theme-option-icon vdb-theme-option-icon--seasonal">
              <SceneIcon scene={autoScene.key} />
            </span>
            <span className="vdb-theme-option-copy">
              <strong>Seasonal · Auto</strong>
              <small>
                {autoScene.label} · {autoScene.regionName}
              </small>
            </span>
            {autoActive && <Check className="vdb-theme-check" />}
          </button>

          <p className="vdb-theme-menu-heading vdb-theme-menu-heading--scenes">
            Preview seasonal scenes
          </p>
          <div className="vdb-theme-scene-grid">
            {SCENE_KEYS.map((key) => {
              const active = theme === 'seasonal' && sceneOverride === key;
              return (
                <button
                  key={key}
                  type="button"
                  role="option"
                  aria-selected={active}
                  aria-label={`Preview ${SCENE_LABELS[key]}`}
                  className={`vdb-theme-option vdb-theme-option--scene${active ? ' is-active' : ''}`}
                  onClick={() => selectScene(key)}
                >
                  <span className={`vdb-theme-scene-thumb vdb-theme-scene-thumb--${key}`}>
                    <SceneArtwork scene={buildSceneFromKey(key, autoScene)} compact />
                  </span>
                  <span className="vdb-theme-option-copy">
                    <strong>{SCENE_LABELS[key]}</strong>
                    <small>Live preview</small>
                  </span>
                  {active && <Check className="vdb-theme-check" />}
                </button>
              );
            })}
          </div>

          <p className="vdb-theme-menu-note">
            Auto follows {autoScene.timeZone} · {autoScene.regionName} · {autoScene.hour}:00
          </p>
        </div>
      )}
    </div>
  );
};
