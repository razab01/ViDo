import '@testing-library/jest-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Tests must never reach the network; UI enrichment falls back to local answers.
// Re-stub before each test because lookup tests intentionally unstub their mocks.
beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('Network disabled in tests'); }));
});

// Vitest + React 19: make sure each test starts with an empty DOM.
afterEach(() => {
  cleanup();
});
