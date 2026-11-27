import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { ThemeProvider } from './ThemeProvider';
import { ThemeSwitcher } from './ThemeSwitcher';
import { SCENE_KEYS, SCENE_LABELS, SCENE_STORAGE_KEY, THEME_STORAGE_KEY } from './theme';

describe('ThemeSwitcher', () => {
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

  it('offers General, Seasonal Auto, and every seasonal scene preview', () => {
    render(
      <ThemeProvider>
        <ThemeSwitcher />
      </ThemeProvider>,
    );

    fireEvent.click(screen.getByRole('button', { name: /general/i }));

    expect(screen.getByRole('option', { name: /general/i })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /seasonal · auto/i })).toBeInTheDocument();
    SCENE_KEYS.forEach((key) => {
      expect(
        screen.getByRole('option', { name: `Preview ${SCENE_LABELS[key]}` }),
      ).toBeInTheDocument();
    });
  });

  it('applies a selected scene preview', () => {
    render(
      <ThemeProvider>
        <ThemeSwitcher />
      </ThemeProvider>,
    );

    fireEvent.click(screen.getByRole('button', { name: /general/i }));
    fireEvent.click(screen.getByRole('option', { name: 'Preview Summer night' }));

    expect(document.documentElement.dataset.theme).toBe('seasonal');
    expect(document.documentElement.dataset.scene).toBe('summer-night');
    expect(document.documentElement.dataset.scenePreview).toBe('true');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('seasonal');
    expect(localStorage.getItem(SCENE_STORAGE_KEY)).toBe('summer-night');
  });
});
