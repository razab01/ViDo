import React, { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Check, Sparkles } from 'lucide-react';
import { SceneArtwork } from './SeasonalBackdrop';
import { useTheme } from './ThemeProvider';
import { buildSceneFromKey, SCENE_KEYS, SCENE_LABELS, type SceneKey } from './theme';

const SCENE_NOTES: Record<SceneKey, string> = {
  'winter-day': 'A slow orbit over wind-carved fast ice and ringed nunataks: emperor adults and chicks huddle near a Weddell seal while an orca surfaces in the meltwater lead under bright, sunless polar light.',
  'winter-night': 'Wide aurora curtains wrap the polar sky, with fine rays, drifting folds, snow, sea ice, and a twelve-second brightness phrase.',
  'summer-day': 'A real beach day: palms, parasol, surf, and a far-off sailboat.',
  'summer-night': 'A moonlit dune field with buttes, cacti, a campfire, and quiet desert wildlife.',
  'rainy-day': 'A winding river cuts through emerald canopy; light shafts fall between the showers and a rainbow opens over the far bank.',
  'rainy-night': 'Fireflies, wet leaves, and a soft moonbow carry the rain through a dark emerald forest.',
};

const SCENE_SEASONS: Record<SceneKey, string> = {
  'winter-day': 'Antarctic / Daylight',
  'winter-night': 'Antarctic / Southern lights',
  'summer-day': 'Coast / Golden hour',
  'summer-night': 'Desert / After dark',
  'rainy-day': 'Rainforest / After the rain',
  'rainy-night': 'Rainforest / Moonbow',
};

const initialSpotlight = (theme: string, sceneKey: SceneKey): SceneKey =>
  theme === 'seasonal' ? sceneKey : 'summer-night';

export const SeasonalGallery: React.FC = () => {
  const { theme, scene, autoScene, sceneOverride, setTheme, setSceneOverride } = useTheme();
  const [spotlightKey, setSpotlightKey] = useState<SceneKey>(() =>
    initialSpotlight(theme, scene.key),
  );
  const stageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (theme === 'seasonal') setSpotlightKey(scene.key);
  }, [theme, scene.key]);

  const spotlightIndex = SCENE_KEYS.indexOf(spotlightKey);
  const spotlightScene = buildSceneFromKey(spotlightKey, autoScene);
  // The desert scene is presented exactly like the reference: no title, hint,
  // button, or label is painted over the artwork.
  const quietStage = spotlightKey === 'summer-night';

  const applyScene = (key: SceneKey) => {
    setTheme('seasonal');
    setSceneOverride(key);
    setSpotlightKey(key);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const element = event.currentTarget;
    const rect = element.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    element.style.setProperty('--spot-x', `${x}%`);
    element.style.setProperty('--spot-y', `${y}%`);
  };

  const handlePointerLeave = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.style.setProperty('--spot-x', '50%');
    event.currentTarget.style.setProperty('--spot-y', '50%');
  };

  return (
    <section className="vdb-gallery" aria-labelledby="vdb-gallery-title">
      <div className="vdb-gallery-head">
        <div>
          <p className="vdb-gallery-kicker">
            <Sparkles aria-hidden="true" /> The seasonal atelier
          </p>
          <h2 id="vdb-gallery-title">Six living worlds. One quiet obsession with detail.</h2>
          <p className="vdb-gallery-intro">
            Every scene is a place, drawn and animated in code — an Antarctic ice shelf, a sunlit
            beach, a rainforest river after the storm. Hover to inspect, click to live inside it.
          </p>
        </div>
        <div className="vdb-gallery-actions" aria-label="Theme mode">
          <button
            type="button"
            className={`vdb-gallery-pill${theme === 'general' ? ' is-active' : ''}`}
            aria-pressed={theme === 'general'}
            onClick={() => setTheme('general')}
          >
            General
          </button>
          <button
            type="button"
            className={`vdb-gallery-pill${
              theme === 'seasonal' && sceneOverride === 'auto' ? ' is-active' : ''
            }`}
            aria-pressed={theme === 'seasonal' && sceneOverride === 'auto'}
            onClick={() => {
              setTheme('seasonal');
              setSceneOverride('auto');
            }}
          >
            Seasonal Auto
          </button>
        </div>
      </div>

      <div
        className="vdb-gallery-stage"
        data-scene={spotlightKey}
        ref={stageRef}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
      >
        <div className="vdb-gallery-stage-orbit" aria-hidden="true" />
        <div className="vdb-gallery-stage-frame">
          <div key={spotlightKey} className="vdb-gallery-stage-art">
            <SceneArtwork scene={spotlightScene} />
          </div>
          {!quietStage && <div className="vdb-gallery-stage-scrim" aria-hidden="true" />}
          {!quietStage && <div className="vdb-gallery-stage-glint" aria-hidden="true" />}

          {!quietStage && (
            <div className="vdb-gallery-stage-top">
              <span className="vdb-gallery-stage-index">
                {String(spotlightIndex + 1).padStart(2, '0')} / 06
              </span>
              <span className="vdb-gallery-stage-live">
                <span aria-hidden="true" /> Live animation
              </span>
            </div>
          )}

          {!quietStage && (
            <div className="vdb-gallery-stage-copy" aria-live="polite">
              <p className="vdb-gallery-stage-season">{SCENE_SEASONS[spotlightKey]}</p>
              <h3>{SCENE_LABELS[spotlightKey]}</h3>
              <p className="vdb-gallery-stage-note">{SCENE_NOTES[spotlightKey]}</p>
              <button
                type="button"
                className="vdb-gallery-stage-apply"
                onClick={() => applyScene(spotlightKey)}
              >
                Apply this scene <ArrowUpRight aria-hidden="true" />
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="vdb-gallery-grid" aria-label="Seasonal scene previews">
        {SCENE_KEYS.map((key, index) => {
          const active = theme === 'seasonal' && sceneOverride === key;
          const cardScene = buildSceneFromKey(key, autoScene);
          return (
            <button
              key={key}
              type="button"
              data-scene={key}
              className={`vdb-gallery-card${active ? ' is-active' : ''}${
                spotlightKey === key ? ' is-spotlit' : ''
              }`}
              style={{ '--i': index } as React.CSSProperties}
              aria-pressed={active}
              aria-label={`Apply ${SCENE_LABELS[key]} theme`}
              onMouseEnter={() => setSpotlightKey(key)}
              onFocus={() => setSpotlightKey(key)}
              onClick={() => applyScene(key)}
            >
              <span className="vdb-gallery-card-index" aria-hidden="true">
                {String(index + 1).padStart(2, '0')}
              </span>
              <span className="vdb-gallery-art">
                <SceneArtwork scene={cardScene} compact />
                <span className="vdb-gallery-card-preview" aria-hidden="true">
                  On stage
                </span>
                {active && (
                  <span className="vdb-gallery-active">
                    <Check aria-hidden="true" /> Applied
                  </span>
                )}
              </span>
              <span className="vdb-gallery-copy">
                <span className="vdb-gallery-label">{SCENE_LABELS[key]}</span>
                <span className="vdb-gallery-note">{SCENE_NOTES[key]}</span>
              </span>
            </button>
          );
        })}
      </div>

      <p className="vdb-gallery-status" aria-live="polite">
        {theme === 'general'
          ? 'General theme is active. Choose a scene above to switch to Seasonal.'
          : `Seasonal theme active · ${scene.label}${scene.isPreview ? ' preview' : ` · ${autoScene.regionName}`}`}
      </p>
    </section>
  );
};
