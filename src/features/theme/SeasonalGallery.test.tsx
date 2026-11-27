import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { SeasonalGallery } from './SeasonalGallery';
import { ThemeProvider } from './ThemeProvider';
import { SCENE_KEYS, SCENE_LABELS, SCENE_STORAGE_KEY, THEME_STORAGE_KEY } from './theme';

describe('SeasonalGallery', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
    const root = document.documentElement;
    delete root.dataset.theme;
    delete root.dataset.season;
    delete root.dataset.daypart;
    delete root.dataset.weather;
    delete root.dataset.scene;
    delete root.dataset.scenePreview;
  });

  it('shows a live preview card for every seasonal scene', () => {
    const { container } = render(
      <ThemeProvider>
        <SeasonalGallery />
      </ThemeProvider>,
    );

    SCENE_KEYS.forEach((key) => {
      expect(
        screen.getByRole('button', { name: `Apply ${SCENE_LABELS[key]} theme` }),
      ).toBeInTheDocument();
    });
    expect(container.querySelectorAll('.vdb-gallery-card .vdb-art')).toHaveLength(
      SCENE_KEYS.length,
    );
    expect(container.querySelector('.vdb-gallery-stage .vdb-art')).toBeInTheDocument();
  });

  it('shows the desert night stage without any written overlay', () => {
    const { container } = render(
      <ThemeProvider>
        <SeasonalGallery />
      </ThemeProvider>,
    );
    const stage = container.querySelector('.vdb-gallery-stage');
    expect(stage).toHaveAttribute('data-scene', 'summer-night');
    expect(stage?.querySelector('.vdb-gallery-stage-copy')).toBeNull();
    expect(stage?.querySelector('.vdb-gallery-stage-top')).toBeNull();
    expect(stage?.querySelector('.vdb-gallery-stage-scrim')).toBeNull();
    expect(stage?.querySelector('.vdb-gallery-stage-glint')).toBeNull();
    expect(stage?.textContent).toBe('');
  });

  it('renders a real-time WebGL canvas for every scene instead of flat vector figures', () => {
    const { container } = render(
      <ThemeProvider>
        <SeasonalGallery />
      </ThemeProvider>,
    );

    expect(container.querySelectorAll('.vdb-gallery-card .vdb-scene-canvas')).toHaveLength(
      SCENE_KEYS.length,
    );
    expect(container.querySelector('.vdb-gallery-stage .vdb-scene-canvas')).toBeInTheDocument();
    container.querySelectorAll('.vdb-gallery .vdb-art').forEach((art) => {
      expect(art.querySelector('canvas')).toBeInTheDocument();
    });
    expect(container.querySelector('.vdb-paper-crane')).toBeNull();
    expect(container.querySelector('.vdb-paper-boat')).toBeNull();
    expect(container.querySelector('.vdb-scene-kid')).toBeNull();
    expect(container.querySelector('.vdb-summer-kid')).toBeNull();
  });

  it('applies a scene when its card is selected', () => {
    render(
      <ThemeProvider>
        <SeasonalGallery />
      </ThemeProvider>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Apply Rainy night theme' }));

    expect(document.documentElement.dataset.theme).toBe('seasonal');
    expect(document.documentElement.dataset.scene).toBe('rainy-night');
    expect(document.documentElement.dataset.scenePreview).toBe('true');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('seasonal');
    expect(localStorage.getItem(SCENE_STORAGE_KEY)).toBe('rainy-night');
  });

  it('updates the spotlight stage and applies from it', () => {
    render(
      <ThemeProvider>
        <SeasonalGallery />
      </ThemeProvider>,
    );

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Apply Winter night theme' }));
    expect(screen.getByRole('heading', { name: 'Winter night' })).toBeInTheDocument();
    expect(document.querySelector('.vdb-gallery-stage')).toHaveAttribute(
      'data-scene',
      'winter-night',
    );

    fireEvent.click(screen.getByRole('button', { name: /apply this scene/i }));
    expect(document.documentElement.dataset.scene).toBe('winter-night');
  });
});
