/**
 * Document loading and lightweight document Q&A for Converse.
 *
 * Plain-text formats are read in the browser so Converse still works offline.
 * PDF and Word files are sent to the optional Python bridge for extraction.
 * Images are deliberately never accepted.
 */
import { extractDocumentWithPython } from '../ml-lab/pythonApi';
import { normalizeQuestion } from './localFallback';

export interface ChatDocument {
  id: string;
  name: string;
  text: string;
  loadedAt: number;
  size: number;
  mime: string;
  truncated?: boolean;
}

export const MAX_DOCUMENT_BYTES = 5 * 1024 * 1024;
export const MAX_DOCUMENT_CHARS = 200_000;

const ALLOWED_EXTENSIONS = new Set([
  '.txt',
  '.md',
  '.markdown',
  '.csv',
  '.json',
  '.pdf',
  '.docx',
]);
const IMAGE_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg', '.avif', '.heic',
  '.heif', '.tif', '.tiff', '.ico',
]);
const PYTHON_EXTENSIONS = new Set(['.pdf', '.docx']);

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with', 'is', 'are',
  'was', 'were', 'be', 'by', 'as', 'at', 'from', 'this', 'that', 'these', 'those',
  'what', 'which', 'who', 'whom', 'how', 'why', 'when', 'do', 'does', 'did', 'look',
  'tell', 'me', 'about', 'it', 'its', 'can', 'you', 'your', 'my', 'into', 'there',
]);

const extensionOf = (name: string): string => {
  const dot = String(name || '').lastIndexOf('.');
  return dot < 0 ? '' : name.slice(dot).toLowerCase();
};

const newId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `doc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

/** Return a user-friendly error, or null when the file is an allowed document. */
export function validateDocumentFile(file: File): string | null {
  const name = String(file?.name || '');
  const extension = extensionOf(name);
  const mime = String(file?.type || '').toLowerCase();

  if (IMAGE_EXTENSIONS.has(extension) || mime.startsWith('image/')) {
    return 'Images are not supported. Load a TXT, MD, CSV, JSON, PDF or Word (.docx) document.';
  }
  if (!ALLOWED_EXTENSIONS.has(extension)) {
    if (extension === '.doc') {
      return 'Legacy .doc files are not supported. Save the file as .docx and load it again.';
    }
    return 'Load a TXT, MD, CSV, JSON, PDF or Word (.docx) document. Images are not supported.';
  }
  if (!file.size) return 'That document is empty.';
  if (file.size > MAX_DOCUMENT_BYTES) return 'That document is too large. The maximum size is 5 MB.';
  return null;
}

/** Read a validated file into persisted chat-document state. */
export async function loadChatDocument(file: File): Promise<ChatDocument> {
  const error = validateDocumentFile(file);
  if (error) throw new Error(error);

  let text = '';
  let truncated = false;
  if (PYTHON_EXTENSIONS.has(extensionOf(file.name))) {
    const extracted = await extractDocumentWithPython(file);
    text = extracted.text;
    truncated = extracted.truncated;
  } else {
    text = await file.text();
  }

  const cleaned = text
    .replace(/\r\n?/g, '\n')
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .trim();
  if (!cleaned) throw new Error('I could not find any readable text in that document.');

  truncated = truncated || cleaned.length > MAX_DOCUMENT_CHARS;
  return {
    id: newId(),
    name: file.name,
    text: cleaned.slice(0, MAX_DOCUMENT_CHARS),
    loadedAt: Date.now(),
    size: file.size,
    mime: file.type || 'application/octet-stream',
    truncated,
  };
}

const tokenize = (value: string): string[] =>
  (normalizeQuestion(value).toLowerCase().match(/[a-z0-9_]+/g) ?? []).filter(
    (token) => token.length > 1 && !STOP_WORDS.has(token)
  );

function chunkText(text: string): string[] {
  return String(text || '')
    .split(/\n\s*\n+/)
    .flatMap((paragraph) => {
      const clean = paragraph.replace(/\s+/g, ' ').trim();
      if (!clean) return [];
      if (clean.length <= 1200) return [clean];
      const pieces: string[] = [];
      for (let index = 0; index < clean.length; index += 1000) {
        const piece = clean.slice(index, index + 1200).trim();
        if (piece) pieces.push(piece);
      }
      return pieces;
    });
}

/** Answer from the loaded document using token-overlap retrieval. */
export function documentAnswer(question: string, document: ChatDocument | null | undefined): string | null {
  if (!document?.text?.trim()) return null;

  const normalized = normalizeQuestion(question);
  const identifiers = normalized.match(/[A-Za-z]*-?\d+/g) ?? [];
  const wanted = Array.from(new Set([...tokenize(normalized), ...identifiers.map((token) => token.toLowerCase())]));
  if (!wanted.length) return null;

  const scored = chunkText(document.text)
    .map((chunk, index) => {
      const chunkTokens = new Set(tokenize(chunk));
      const score = wanted.filter(
        (token) => chunkTokens.has(token) || new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(chunk)
      ).length;
      return { chunk, index, score };
    })
    .filter((item) => item.score > 0)
    .sort((first, second) => second.score - first.score || first.index - second.index)
    .slice(0, 2);

  if (!scored.length) return null;
  const excerpt = scored.map((item) => item.chunk).join(' … ');
  const trimmed = excerpt.length > 1600 ? `${excerpt.slice(0, 1600).trimEnd()}…` : excerpt;
  return `From "${document.name}": ${trimmed}`;
}
