import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { createSceneWorld } from './scene3d';
import { useTheme } from './ThemeProvider';
import type { SeasonalScene } from './theme';

export interface SceneArtworkProps {
  scene: SeasonalScene;
  className?: string;
  compact?: boolean;
}

/** Real-time WebGL scene used by the backdrop, picker, gallery, and landing hero. */
export const SceneArtwork: React.FC<SceneArtworkProps> = ({
  scene,
  className = '',
  compact = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // jsdom and browsers without WebGL should take the lightweight fallback
    // without letting Three.js log a context-creation error on every mount.
    if (!('WebGLRenderingContext' in window) && !('WebGL2RenderingContext' in window)) {
      setSupported(false);
      return;
    }

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: !compact,
        alpha: false,
        powerPreference: compact ? 'default' : 'high-performance',
      });
    } catch {
      setSupported(false);
      return;
    }

    setSupported(true);
    const desertNight = scene.key === 'summer-night';
    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, desertNight ? (compact ? 1.4 : 2) : compact ? 1 : 1.6),
    );
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = desertNight ? 1.2 : scene.daypart === 'night' ? 1.05 : 1.18;
    renderer.shadowMap.enabled = !compact;
    renderer.shadowMap.type = desertNight ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;

    const world = createSceneWorld(scene.key, compact);
    let lastFrame = performance.now();
    let elapsed = 0;
    let frame = 0;
    let visible = true;
    let disposed = false;
    const motionQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    let reduced = motionQuery?.matches ?? false;

    const resize = () => {
      const width = canvas.clientWidth || 1;
      const height = canvas.clientHeight || 1;
      renderer.setSize(width, height, false);
      world.resize(width, height);
      renderer.render(world.scene, world.camera);
    };

    const tick = () => {
      if (disposed) return;
      const now = performance.now();
      const delta = Math.min((now - lastFrame) / 1000, 0.05);
      lastFrame = now;
      if (!reduced) {
        elapsed += delta;
        world.update(elapsed, delta);
      }
      renderer.render(world.scene, world.camera);
      if (!reduced && visible && document.visibilityState !== 'hidden') {
        frame = window.requestAnimationFrame(tick);
      }
    };

    const start = () => {
      if (disposed || reduced || frame) return;
      lastFrame = performance.now();
      frame = window.requestAnimationFrame(tick);
    };

    const stop = () => {
      if (!frame) return;
      window.cancelAnimationFrame(frame);
      frame = 0;
    };

    const handleMotion = (event: MediaQueryListEvent) => {
      reduced = event.matches;
      stop();
      if (reduced) resize();
      else start();
    };

    const handleVisibility = () => {
      if (document.visibilityState === 'hidden') stop();
      else start();
    };

    const resizeObserver =
      typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : undefined;
    resizeObserver?.observe(canvas);

    const intersectionObserver =
      typeof IntersectionObserver !== 'undefined'
        ? new IntersectionObserver(
            ([entry]) => {
              visible = entry?.isIntersecting ?? true;
              if (visible) start();
              else stop();
            },
            { rootMargin: '120px' },
          )
        : undefined;
    intersectionObserver?.observe(canvas);

    motionQuery?.addEventListener?.('change', handleMotion);
    document.addEventListener('visibilitychange', handleVisibility);
    window.addEventListener('resize', resize);
    resize();
    if (!reduced) start();

    return () => {
      disposed = true;
      stop();
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();
      motionQuery?.removeEventListener?.('change', handleMotion);
      document.removeEventListener('visibilitychange', handleVisibility);
      window.removeEventListener('resize', resize);
      world.dispose();
      renderer.dispose();
      // StrictMode and scene changes reuse the same canvas. Forcing loss there
      // would poison the renderer created immediately afterwards.
      if (!canvas.isConnected) renderer.forceContextLoss();
    };
  }, [scene.key, scene.daypart, compact]);

  const classes = [
    'vdb-art',
    `vdb-art--${scene.key}`,
    scene.daypart === 'night' ? 'vdb-art--night' : 'vdb-art--day',
    compact ? 'vdb-art--compact' : '',
    supported ? 'vdb-art--webgl' : 'vdb-art--fallback',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={classes} data-scene={scene.key} data-webgl={supported ? 'true' : 'false'} aria-hidden="true">
      <canvas
        ref={canvasRef}
        className="vdb-scene-canvas"
        style={supported ? undefined : { display: 'none' }}
      />
      {!supported && <div className="vdb-scene-fallback" />}
    </div>
  );
};

export const SeasonalBackdrop: React.FC = () => {
  const { theme, scene } = useTheme();

  if (theme !== 'seasonal') return null;

  return (
    <div className="vdb-season" aria-hidden="true">
      <SceneArtwork scene={scene} />
    </div>
  );
};
