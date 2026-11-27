export type ThemeChoice = 'general' | 'seasonal';
export type DayPart = 'day' | 'night';
export type SeasonKind = 'winter' | 'summer' | 'rainy';
export type SceneKey = `${SeasonKind}-${DayPart}`;
export type SceneOverride = SceneKey | 'auto';

export const SCENE_KEYS: SceneKey[] = [
  'winter-day',
  'winter-night',
  'summer-day',
  'summer-night',
  'rainy-day',
  'rainy-night',
];

export const SCENE_STORAGE_KEY = 'vido-season-scene';

export interface SeasonalScene {
  key: SceneKey;
  season: SeasonKind;
  daypart: DayPart;
  country: string;
  regionName: string;
  timeZone: string;
  hour: number;
  month: number;
  label: string;
  caption: string;
  isPreview?: boolean;
}

export const THEME_STORAGE_KEY = 'vido-theme';

const SOUTHERN_COUNTRIES = new Set([
  'AU', 'NZ', 'ZA', 'AR', 'CL', 'UY', 'PY', 'BO', 'PE', 'BR',
  'MW', 'ZM', 'ZW', 'MZ', 'MG', 'TZ', 'KE', 'UG', 'ID', 'PG',
]);

const TIMEZONE_COUNTRY: Record<string, string> = {
  'Asia/Kolkata': 'IN',
  'Asia/Dhaka': 'BD',
  'Asia/Karachi': 'PK',
  'Asia/Kathmandu': 'NP',
  'Asia/Colombo': 'LK',
  'Asia/Yangon': 'MM',
  'Asia/Bangkok': 'TH',
  'Asia/Ho_Chi_Minh': 'VN',
  'Asia/Manila': 'PH',
  'Asia/Jakarta': 'ID',
  'Asia/Kuala_Lumpur': 'MY',
  'Asia/Singapore': 'SG',
  'Asia/Phnom_Penh': 'KH',
  'Asia/Vientiane': 'LA',
  'Australia/Sydney': 'AU',
  'Australia/Melbourne': 'AU',
  'Australia/Brisbane': 'AU',
  'Australia/Perth': 'AU',
  'Pacific/Auckland': 'NZ',
  'Africa/Johannesburg': 'ZA',
  'America/Argentina/Buenos_Aires': 'AR',
  'America/Santiago': 'CL',
  'America/Sao_Paulo': 'BR',
  'America/New_York': 'US',
  'America/Chicago': 'US',
  'America/Denver': 'US',
  'America/Los_Angeles': 'US',
  'Europe/London': 'GB',
  'Europe/Paris': 'FR',
  'Europe/Berlin': 'DE',
};

const RAINY_MONTHS: Record<string, number[]> = {
  IN: [6, 7, 8, 9],
  BD: [6, 7, 8, 9],
  PK: [7, 8],
  NP: [6, 7, 8, 9],
  LK: [5, 6, 10, 11],
  MM: [6, 7, 8, 9],
  TH: [5, 6, 7, 8, 9, 10],
  VN: [5, 6, 7, 8, 9, 10],
  PH: [6, 7, 8, 9],
  ID: [11, 12, 1, 2, 3],
  MY: [11, 12, 1, 2, 3],
  SG: [11, 12, 1, 2, 3],
  KH: [5, 6, 7, 8, 9, 10],
  LA: [5, 6, 7, 8, 9, 10],
  GH: [4, 5, 6, 7, 8, 9],
  NG: [4, 5, 6, 7, 8, 9, 10],
  BR: [1, 2, 3, 4, 11, 12],
};

const SCENE_COPY: Record<SceneKey, { label: string; verb: string }> = {
  'winter-day': { label: 'Winter day', verb: 'Bright polar daylight over fast ice, an emperor colony, a Weddell seal, and ringed nunataks' },
  'winter-night': { label: 'Winter night', verb: 'Aurora australis above Antarctic ice' },
  'summer-day': { label: 'Summer day', verb: 'A sunlit beach with palms and a sailboat' },
  'summer-night': { label: 'Summer night', verb: 'A silver desert after dark, with a campfire and its nocturnal wildlife' },
  'rainy-day': { label: 'Rainy day', verb: 'A rainforest river after the storm' },
  'rainy-night': { label: 'Rainy night', verb: 'A moonlit rainforest river with fireflies and a moonbow' },
};

export const SCENE_LABELS: Record<SceneKey, string> = {
  'winter-day': SCENE_COPY['winter-day'].label,
  'winter-night': SCENE_COPY['winter-night'].label,
  'summer-day': SCENE_COPY['summer-day'].label,
  'summer-night': SCENE_COPY['summer-night'].label,
  'rainy-day': SCENE_COPY['rainy-day'].label,
  'rainy-night': SCENE_COPY['rainy-night'].label,
};

function countryFromLanguage(language?: string): string {
  if (!language) return '';
  const parts = language.replace('_', '-').split('-');
  const candidate = parts[parts.length - 1];
  return /^[A-Za-z]{2}$/.test(candidate) ? candidate.toUpperCase() : '';
}

function countryFromTimeZone(timeZone: string): string {
  if (TIMEZONE_COUNTRY[timeZone]) return TIMEZONE_COUNTRY[timeZone];
  const [area, city] = timeZone.split('/');
  if (area === 'Australia') return 'AU';
  if (timeZone.startsWith('Pacific/Auckland')) return 'NZ';
  if (city === 'Johannesburg') return 'ZA';
  return '';
}

function regionNameFor(country: string): string {
  if (!country) return 'your region';
  try {
    return new Intl.DisplayNames(['en'], { type: 'region' }).of(country) ?? country;
  } catch {
    return country;
  }
}

function isSouthern(country: string, timeZone: string): boolean {
  if (SOUTHERN_COUNTRIES.has(country)) return true;
  return (
    timeZone.startsWith('Australia/') ||
    timeZone.startsWith('Pacific/Auckland') ||
    timeZone.startsWith('America/Argentina') ||
    timeZone === 'America/Santiago' ||
    timeZone === 'America/Sao_Paulo' ||
    timeZone === 'Africa/Johannesburg'
  );
}

function resolveSeason(country: string, month: number, southern: boolean): SeasonKind {
  if (RAINY_MONTHS[country]?.includes(month)) return 'rainy';
  const winterMonths = southern ? [6, 7, 8] : [12, 1, 2];
  const summerMonths = southern ? [12, 1, 2] : [6, 7, 8];
  if (winterMonths.includes(month)) return 'winter';
  if (summerMonths.includes(month)) return 'summer';
  // Spring and autumn stay warm and bright in the seasonal theme.
  return 'summer';
}

export function getSeasonalScene(
  date: Date = new Date(),
  options: { timeZone?: string; country?: string; language?: string } = {},
): SeasonalScene {
  const timeZone =
    options.timeZone ||
    (() => {
      try {
        return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
      } catch {
        return 'UTC';
      }
    })();

  const country =
    options.country ||
    countryFromLanguage(options.language ?? (typeof navigator !== 'undefined' ? navigator.language : undefined)) ||
    countryFromTimeZone(timeZone) ||
    'US';

  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: 'numeric',
    month: 'numeric',
    hour12: false,
  }).formatToParts(date);

  const hourPart = parts.find((part) => part.type === 'hour')?.value ?? '0';
  const monthPart = parts.find((part) => part.type === 'month')?.value ?? '1';
  const hour = Number(hourPart) % 24;
  const month = Number(monthPart);
  const daypart: DayPart = hour >= 6 && hour < 18 ? 'day' : 'night';
  const season = resolveSeason(country, month, isSouthern(country, timeZone));
  const key = `${season}-${daypart}` as SceneKey;
  const copy = SCENE_COPY[key];
  const regionName = regionNameFor(country);

  return {
    key,
    season,
    daypart,
    country,
    regionName,
    timeZone,
    hour,
    month,
    label: copy.label,
    caption: `${copy.verb} · ${regionName}`,
  };
}

export function readSceneOverride(): SceneOverride {
  try {
    const stored = localStorage.getItem(SCENE_STORAGE_KEY);
    return stored && (SCENE_KEYS as string[]).includes(stored) ? (stored as SceneKey) : 'auto';
  } catch {
    return 'auto';
  }
}

export function writeSceneOverride(choice: SceneOverride): void {
  try {
    if (choice === 'auto') {
      localStorage.removeItem(SCENE_STORAGE_KEY);
    } else {
      localStorage.setItem(SCENE_STORAGE_KEY, choice);
    }
  } catch {
    // Storage can be unavailable in private mode; the in-memory choice still works.
  }
}

export function buildSceneFromKey(
  key: SceneKey,
  base: SeasonalScene = getSeasonalScene(),
): SeasonalScene {
  const [season, daypart] = key.split('-') as [SeasonKind, DayPart];
  return {
    ...base,
    key,
    season,
    daypart,
    label: SCENE_LABELS[key],
    caption: `Preview: ${SCENE_COPY[key].verb}`,
    isPreview: true,
  };
}

export function readThemeChoice(): ThemeChoice {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    return stored === 'seasonal' ? 'seasonal' : 'general';
  } catch {
    return 'general';
  }
}

export function writeThemeChoice(choice: ThemeChoice): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, choice);
  } catch {
    // Storage can be unavailable in private mode; the in-memory choice still works.
  }
}

export function applyThemeAttributes(choice: ThemeChoice, scene: SeasonalScene): void {
  const root = document.documentElement;
  root.dataset.theme = choice;

  if (choice === 'seasonal') {
    root.dataset.season = scene.season;
    root.dataset.daypart = scene.daypart;
    root.dataset.weather = scene.season === 'rainy' ? 'rain' : 'clear';
    root.dataset.scene = scene.key;
    if (scene.isPreview) {
      root.dataset.scenePreview = 'true';
    } else {
      delete root.dataset.scenePreview;
    }
    root.style.colorScheme = scene.daypart === 'night' ? 'dark' : 'light';
  } else {
    delete root.dataset.season;
    delete root.dataset.daypart;
    delete root.dataset.weather;
    delete root.dataset.scene;
    delete root.dataset.scenePreview;
    root.style.colorScheme = 'light';
  }
}
