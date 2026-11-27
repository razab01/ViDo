import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  applyThemeAttributes,
  buildSceneFromKey,
  getSeasonalScene,
  readSceneOverride,
  readThemeChoice,
  SCENE_KEYS,
  SCENE_STORAGE_KEY,
  writeSceneOverride,
  writeThemeChoice,
} from './theme';

describe('getSeasonalScene', () => {
  it('uses northern winter in day time', () => {
    const scene = getSeasonalScene(new Date('2026-01-15T12:00:00Z'), {
      timeZone: 'Asia/Kolkata',
      country: 'IN',
    });
    expect(scene.key).toBe('winter-day');
    expect(scene.season).toBe('winter');
    expect(scene.daypart).toBe('day');
  });

  it('uses northern summer at night', () => {
    const scene = getSeasonalScene(new Date('2026-07-15T04:00:00Z'), {
      timeZone: 'America/New_York',
      country: 'US',
    });
    expect(scene.key).toBe('summer-night');
    expect(scene.daypart).toBe('night');
  });

  it('flips seasons in the southern hemisphere', () => {
    const scene = getSeasonalScene(new Date('2026-07-15T04:00:00Z'), {
      timeZone: 'Australia/Sydney',
      country: 'AU',
    });
    expect(scene.key).toBe('winter-day');
  });

  it('uses the monsoon window for rainy day time', () => {
    const scene = getSeasonalScene(new Date('2026-07-15T06:00:00Z'), {
      timeZone: 'Asia/Kolkata',
      country: 'IN',
    });
    expect(scene.key).toBe('rainy-day');
    expect(scene.caption).toContain('A rainforest river after the storm');
  });

  it('uses the monsoon window for rainy night time', () => {
    const scene = getSeasonalScene(new Date('2026-07-15T18:00:00Z'), {
      timeZone: 'Asia/Kolkata',
      country: 'IN',
    });
    expect(scene.key).toBe('rainy-night');
    expect(scene.caption).toContain('A moonlit rainforest river with fireflies and a moonbow');
  });

  it('uses summer day time outside monsoon countries', () => {
    const scene = getSeasonalScene(new Date('2026-07-15T16:00:00Z'), {
      timeZone: 'America/New_York',
      country: 'US',
    });
    expect(scene.key).toBe('summer-day');
  });
});

describe('manual scene preview', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
    const root = document.documentElement;
    delete root.dataset.theme;
    delete root.dataset.season;
    delete root.dataset.daypart;
    delete root.dataset.weather;
    delete root.dataset.scene;
    delete root.dataset.scenePreview;
  });

  it('defaults to auto for a missing or invalid override', () => {
    expect(readSceneOverride()).toBe('auto');
    localStorage.setItem(SCENE_STORAGE_KEY, 'spring-day');
    expect(readSceneOverride()).toBe('auto');
  });

  it('persists a chosen scene and clears it with auto', () => {
    writeSceneOverride('summer-night');
    expect(readSceneOverride()).toBe('summer-night');
    expect(localStorage.getItem(SCENE_STORAGE_KEY)).toBe('summer-night');

    writeSceneOverride('auto');
    expect(readSceneOverride()).toBe('auto');
    expect(localStorage.getItem(SCENE_STORAGE_KEY)).toBeNull();
  });

  it('builds a marked preview scene without changing location metadata', () => {
    const base = getSeasonalScene(new Date('2026-01-15T12:00:00Z'), {
      timeZone: 'Asia/Kolkata',
      country: 'IN',
    });
    const preview = buildSceneFromKey('summer-night', base);

    expect(preview.key).toBe('summer-night');
    expect(preview.season).toBe('summer');
    expect(preview.daypart).toBe('night');
    expect(preview.timeZone).toBe(base.timeZone);
    expect(preview.country).toBe(base.country);
    expect(preview.isPreview).toBe(true);
    expect(preview.caption).toContain('Preview');
  });

  it('can preview all six seasonal scenes', () => {
    expect(SCENE_KEYS).toHaveLength(6);
    SCENE_KEYS.forEach((key) => {
      const preview = buildSceneFromKey(key);
      expect(preview.key).toBe(key);
      expect(preview.isPreview).toBe(true);

      applyThemeAttributes('seasonal', preview);
      expect(document.documentElement.dataset.scene).toBe(key);
      expect(document.documentElement.dataset.scenePreview).toBe('true');
    });
  });

  it('keeps theme choice persistence separate from scene preview', () => {
    writeThemeChoice('seasonal');
    writeSceneOverride('rainy-night');
    expect(readThemeChoice()).toBe('seasonal');
    expect(readSceneOverride()).toBe('rainy-night');
  });
});
