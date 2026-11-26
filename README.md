<div align="center">

# ViDo

### See the work. Share the progress. Deliver together.

A modern task-management workspace built for focused individuals and transparent teams.

[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev/)
[![React](https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![NestJS](https://img.shields.io/badge/NestJS_11-E0234E?style=for-the-badge&logo=nestjs&logoColor=white)](https://nestjs.com/)
[![Prisma](https://img.shields.io/badge/Prisma_6-2D3748?style=for-the-badge&logo=prisma&logoColor=white)](https://www.prisma.io/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS_v4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Vitest](https://img.shields.io/badge/Vitest-729B1B?style=for-the-badge&logo=vitest&logoColor=white)](https://vitest.dev/)

</div>

---

## 💡 What is ViDo?

ViDo is a **task-management workspace** designed to make work progress visible and verifiable without noise.

Instead of relying on constant status meetings or ambiguous binary to-do lists, ViDo tracks **incremental checkpoints** and **timestamped progress**. It is built on the philosophy that tracking *what was done, when, and by whom* is fundamentally more valuable than simply knowing *if* a task is completed.

### Why ViDo?

- **Clarity over complexity**: A calm, focused interface that highlights what matters now.
- **Progress visibility**: Update status continuously (e.g., 20%, 45%, 75%) rather than waiting for final completion.
- **Accountability**: Lightweight checkpoint history that proves work without the burden of micro-reporting.
- **Deadline awareness**: Unified views for what is due Today, Upcoming, or Needs Attention.
- **Planning and analytics**: A planning calendar and completion analytics with editable weekly/monthly targets.
- **Zero-friction local testing**: Full workspace behavior on the local Firebase emulator suite with no real credentials required.

> **MVP Status**: The application includes a fully functional public landing page, a shareable preview demo (`/preview`), live Firebase emulator authentication with per-user calendar/planning data, collaborative team workspaces, and a complete task/checkpoint lifecycle.

---

## 👥 Who ViDo is For

<table>
<tr>
<td width="50%" valign="top">

### 🧑‍💻 Individual Contributors

- Keep focused priorities in one place
- Update progress incrementally as work advances
- Never miss deadlines with timeline-based views
- Track completion trends with weekly/monthly analytics
- Build a verifiable record of work

</td>
<td width="50%" valign="top">

### 👔 Team Leaders & Managers

- Organize focused workspaces
- Clear assignment and ownership
- Follow progress and checkpoint updates
- Flag late deliveries automatically
- Coordinated history without redundant meetings

</td>
</tr>
</table>

---

## 🗺️ Application Routes

| Route | Description |
|---|---|
| `/` | Public landing page with preview, design principles, and sign-in flow. |
| `/workspace/*` | Authenticated workspace (Firebase Auth `:9099`, Firestore `:8080`, `demo-vido` project). |
| `/preview/*` | Full guest demo with deterministic sample data and browser-local planning state. |
| `/preview\|workspace/tasks` | Filterable task management view for active responsibilities. |
| `/preview\|workspace/today`, `/upcoming`, `/needs-attention`, `/completed`, `/archived` | Section-specific task lists derived from shared filters. |
| `/preview\|workspace/calendar` | Weekly/monthly planning calendar combining ViDo tasks and calendar events. |
| `/preview\|workspace/analytics` | Completion analytics with editable weekly/monthly targets and cumulative progress. |
| `/preview\|workspace/team/:teamId` | Shared team workspace with milestones, member roster, and collaborative task backlog (demo data in preview). |
| `/workspace/team/:teamId` | Authenticated team workspace with lead permissions, member assignments, and progress tracking. |
| `/preview/chat-lab` | Offline-capable conversational lab for questions, task creation, flowcharts, and document context. |
| `/style-lab` | Interactive sandbox for `Cobalt Blue`, `Deep Indigo`, and `Forest Green` accent palettes. |

---

## ✨ Core Features

### 🔐 Authentication & Session Management
- **Google sign-in (emulator)**: Popup-based sign-in powered by Firebase Authentication against the local Auth emulator.
- **No real credentials required**: The default `demo-vido` project and emulator suite run entirely on your machine.
- **Per-user planning data**: Calendar events and weekly/monthly targets are stored in Firestore under `users/{uid}` for signed-in users.
- **Zero-barrier demo**: `/preview/*` is fully interactive without an account.

### 👥 Team Workspaces & Shared Coordination
- **Unified Sidebar Navigation**: Personal views (*Dashboard*, *Tasks*, *Calendar*, *Analytics*) alongside active collaborative spaces under `Teams & Shared Projects`.
- **Email-Based Invitations**: Team Leads invite collaborators by Google email address with duplicate checking and case-insensitive matching.
- **In-App Notification Center**: Top utility bar notification bell (`🔔`) alerts invited users in real-time, providing one-click **Accept** and **Reject** actions.
- **Phased Milestone Roadmap**: Common project goals broken down into sequential delivery phases with target dates and status badges (`Completed`, `Active Phase`, `Upcoming`).
- **Lead-Driven Permissions**: Team Leads administer team settings, milestones, and task assignments. Assignees maintain ownership over their progress (0–100%) and checkpoint logs. Tasks are read-only for non-assigned teammates.

### ✅ Task Lifecycle & Checkpoints
- **Incremental progress tracking**: Measure work continuously rather than relying on binary done/not-done toggles.
- **Timestamped checkpoints**: Keep a chronological audit trail of updates for every responsibility.
- **Deadline-aware views**: Shared filters power Today, Upcoming, Needs Attention, Completed, and Archived sections.

### 🗓️ Planning Calendar
- Unified **weekly** and **monthly** layouts showing ViDo tasks and calendar commitments together.
- Create and manage planning calendar events alongside tasks in weekly and monthly views.
- **Task ↔ event flows**: add a task to the calendar as a linked event, import an event as a task, or create/remove manual events.
- Live events are stored at `users/{uid}/calendarEvents`; preview events are stored in `localStorage`.

### 📊 Completion Analytics
- Weekly completion percentage and progress against a user-set weekly target.
- Monthly goal tracking with cumulative totals contributed by each week.
- Editable targets persisted per user in Firestore (live) or in `localStorage` (preview), with progress bars and roll-up charts.

### 💬 Chat Lab

Chat Lab is a separate conversational surface (`/preview/chat-lab`) that keeps its own thread history, task draft, last created task, and loaded document in browser storage. It stays independent from ML Lab but can reuse the user's uploaded dataset when available.

**Message flow, in order:**

1. **Normalise and resolve** — conversational fillers are stripped, common typos are corrected, and pronouns such as `he`, `she`, or `it` are resolved against earlier turns.
2. **Special intents first** — `process graph for this` reuses the last task created in chat; task-creation messages parse titles, relative dates/times, ask for a flowchart when needed, collect steps, and call the workspace task creator.
3. **Python-first answers** — when the optional Python service is running, questions go through maths/SymPy, curated smart answers, world facts, loaded-document context, subject classification, applied Math/Physics/Chemistry solvers, pilot subject cards, Wikipedia topic selection, and dataset Graph-RAG. Wikipedia evidence can be checked in parallel without adding a badge to the reply.
4. **Browser fallback** — if Python is unavailable, the same common paths run locally: arithmetic (Shunting Yard), maths concepts, dataset answers, document answers, and Wikipedia lookups.
5. **Render cleanly** — answers are shown without engine/source badges, and flowcharts render inline as SVG.

**Task examples understood by chat:**

- `hey create a task for tomorrow morning 4 am for walk.` → title `walk`, due tomorrow `04:00`, start today, then a flowchart prompt.
- `yes, steps: plan route, pack shoes, go for walk` → creates the workspace task with inline flowchart steps.
- `ok can u show me a process graph for this` → attaches a flowchart to the most recently created chat task.

**Document context:**

- **Attach document** accepts TXT, MD, CSV, JSON, PDF, and DOCX. PDF/DOCX extraction uses the Python bridge; text formats are read in-browser.
- Images and legacy `.doc` files are rejected with a clear message. Files are capped at 5 MB and extracted text at 200,000 characters.
- An attached file is posted into the conversation as an attachment card and stays active for follow-up questions in that thread. Questions are answered from the document before generic Wikipedia/dataset fallback.

**Pilot subject cards (Python-only):**

- The first ten subjects are Mathematics, Physics, Chemistry, Biology, Economics, Computer Science, History, Geography, Politics and Law, and Psychology.
- Each subject has 50 curated topic cards in `python/subject_data/`; one shared router classifies the subject and topic from aliases, subject signals, and exact phrase matches.
- High-confidence card answers stay offline. Wikipedia remains the fallback for topics outside the pilot set, and the browser fallback is unchanged in this phase.
- Basic questions get the concise card answer. Questions that ask for an advanced/graduate explanation use `python/subject_advanced.py` to expand the same card into definition, formal core, mechanism, framing and limits, with topic-specific notes for the 250 advanced sweep topics. The advanced route runs before short curated answers so deep questions such as Schur complements and nilpotent matrices get the full explanation, while loaded documents still win for document questions.

**Applied solvers and evaluation (Python-only):**

- Numeric questions routed to Mathematics, Physics, or Chemistry are solved before the definition cards answer: equations/simplification, kinematics, Newton's second law, work/energy/power, ideal gas law, Ohm's law, molar mass/moles, pH, and stoichiometry ratios.
- Wikipedia evidence checks record whether an article supports a card, solver, or fallback answer. The checks are internal metadata only; chat replies stay free of `Python`, `Local`, `Wikipedia`, and other engine/source badges.
- `python python/evaluation.py` runs a scored offline sweep across concept, solver, task, document, and out-of-scope questions. The default corpus is local, so evaluation never needs the network.
- `npm run sweep:python` generates a deterministic 1,000-question breadth sweep (500 Basic, 200 Intermediate, 250 Advanced, 50 Graduate). Every question and exact response is written to `python/chat_sweep_report.md` and `python/chat_sweep_report.jsonl`, together with the passing unittest/evaluation names and per-level counts. `npm run xlsx:python` converts that JSONL into a formatted Excel workbook at `python/chat_sweep_report.xlsx` plus a CSV copy. If Excel has the workbook open, the converter writes `python/chat_sweep_report_v2.xlsx` instead of failing.

---

### 🎨 Living Design System
- Desktop-first, contrast-compliant UI with semantic CSS custom properties.
- Three live accent palettes in `/style-lab`.

---

## ⚡ Quick Start for Developers

### Prerequisites

- **Node.js 18+** (Node 20+ LTS recommended)
- **npm** (comes with Node)
- **Java 11+** (required by the Firebase Emulator Suite) — verify with `java -version`
- **Firebase CLI** (recommended): `npm install -g firebase-tools`

### Installation & Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vicharanashala/ViDo.git
   cd ViDo
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the local development environment:**
   - **Full Stack (Frontend + NestJS Backend API)**:
     ```bash
     npm run dev:all
     ```
     *(Runs Vite on `http://localhost:5173` and NestJS on `http://localhost:3001` with SQLite `dev.db`)*
   - **Frontend only**:
     ```bash
     npm run dev
     ```
     *(Runs Vite; in frontend-only mode team data can use the mock repository/dev bridge, and Firebase emulators may be auto-started when available.)*
   - **Frontend + Python answer engine**:
     ```bash
     pip install -r python/requirements.txt
     npm run dev:python
     ```
     *(Enables Python-first maths, Graph-RAG, smart answers, and PDF/DOCX extraction. Chat Lab still works without it via the browser fallback.)*

4. **(Optional) Start the full emulator suite manually in a separate terminal:**
   ```bash
   npm run emu
   ```

5. **Open the app:**
   Navigate to [http://localhost:5173](http://localhost:5173).

> [!TIP]
> When you run `npm run dev` or `npm run dev:all`, Vite will attempt to start missing Firebase emulators when Java and the Firebase CLI are available.

---

## 🔐 Firebase Emulator Setup

ViDo is configured for **zero-friction local testing**. No Google Cloud account, credit card, or live Firebase credentials are required.

- **Auth emulator**: `127.0.0.1:9099`
- **Firestore emulator**: `127.0.0.1:8080`
- **Emulator UI**: `127.0.0.1:4000` (when running `npm run emu` / `npm run emulators`)
- **Project ID**: `demo-vido`

Configuration lives in [`.env.example`](./.env.example), [`firebase.json`](./firebase.json), [`firestore.rules`](./firestore.rules), and [`firestore.indexes.json`](./firestore.indexes.json).

### Interactive Sign-In Flow

1. Open `http://localhost:5173`.
2. Click **Continue with Google** / **Sign In to Dashboard**.
3. Choose any mock account in the Firebase Auth emulator popup.
4. You are taken to `/workspace` with a per-user workspace.
5. Use **Calendar** to add/import events and manage planning items; use **Analytics** to adjust weekly/monthly targets.
6. Click **Sign Out** to return to the landing page.

---

## 👥 Collaborative Team Spaces & Multi-Backend Architecture

ViDo supports multiple backend modes designed for zero-friction local testing as well as production scalability:

### 1. Dedicated NestJS Backend with Prisma ORM & SSE (Recommended Local API)
For full-stack developers who want a dedicated TypeScript API without any cloud dependency:
- **Zero Cloud Setup**: Uses Prisma ORM with a local SQLite database (`server/prisma/dev.db`).
- **Concurrent Startup**: Run `npm run dev:all` to launch the Vite frontend (:5173) and NestJS backend (:3001) together.
- **Real-Time Push**: Server-Sent Events (SSE) broadcast team invitations and task updates in real-time across tabs and windows.
- **Firebase AuthGuard**: Verifies Firebase ID tokens from both the local Firebase Emulator (`127.0.0.1:9099`) and live Firebase.
- **Production Switching**: Seamlessly switches from SQLite to PostgreSQL by configuring `DATABASE_URL` in production.

### 2. Offline Mock Repository & Vite Sync Bridge (Frontend-Only)
When running frontend-only without the NestJS backend:
- **Zero Configuration**: Uses `MockTeamRepository` with the Vite dev server SSE sync bridge.
- **Preloaded Personas & Data**: Test immediately with preconfigured teams, phased milestones, and simulated invitations.

### 3. Optional Production Supabase Integration
For teams preferring a managed PostgreSQL BaaS:
1. Create a Supabase project at [supabase.com](https://supabase.com).
2. Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in `.env.local`.
3. Restart Vite (`npm run dev`). ViDo's `getTeamRepository()` auto-detects credentials and connects to Supabase PostgreSQL and Realtime channels.

---

## 🧪 Verification & Scripts

| Command | Description |
|---|---|
| `npm run dev:all` | Starts Vite frontend (:5173) and NestJS backend (:3001) concurrently. |
| `npm run dev` | Starts the Vite dev server (and attempts to auto-start missing Firebase emulators when available). |
| `npm run dev:server` | Starts the NestJS backend with live reload (`ts-node-dev`). |
| `npm run build:server` | Compiles the NestJS backend into `server/dist/`. |
| `npm test` | Runs the Vitest suite. |
| `npm run test:watch` | Runs Vitest in interactive watch mode. |
| `npm run typecheck` | Validates TypeScript types across the frontend (`tsc -b --noEmit`). |
| `npm run build` | Compiles an optimized production frontend build into `dist/`. |
| `npm run preview` | Locally previews the compiled production build from `dist/`. |
| `npm run python:api` | Starts the optional Python ML/Graph-RAG API on `127.0.0.1:8000`. |
| `npm run dev:python` / `npm run preview:python` | Runs the frontend while Vite auto-starts the bundled Python API on `127.0.0.1:8000` for the full Chat Lab answer path. |
| `npm run test:python` | Runs the Python answer-engine checks. |
| `npm run eval:python` | Runs the scored offline answer sweep and Wikipedia evidence checks. |
| `npm run sweep:python` | Generates the 1,000-question Chat Lab response report with per-level pass/fail counts. |
| `npm run xlsx:python` | Converts the sweep report into `python/chat_sweep_report.xlsx` and a CSV copy. |
| `npm run emulators` / `npm run emu` | Starts the Firebase emulator suite for `demo-vido` (Auth + Firestore + UI). |
| `npm run emu:export` | Exports emulator data to `./firebase-export-local`. |
| `npm run emu:import` | Starts emulators importing/exporting local data. |

### Mandatory Pre-Commit Check

```bash
npm run test:python && npm run eval:python && npm run typecheck && npm test && npm run build
```

---

## 🗂️ Architecture & Code Organization

ViDo follows a **feature-driven, modular frontend architecture**. For the complete directory tree and placement rules, see **[FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md)**.

### Key Boundaries

- **UI Primitives (`src/components/ui/`)**: Reusable, accessible, domain-agnostic components (`Button`, `Card`, `Badge`, `Input`, `NoticeModal`). They never import domain models.
- **Feature Slices (`src/features/`)**: Vertical slices for `landing/`, `auth/`, `workspace/`, `teams/`, and `style-lab/`.
- **Data & Pure Logic (`src/data/`)**: Types, deterministic fixtures, filters, calendar helpers, and analytics calculations.
- **Firebase & Persistence (`src/lib/`)**: Firebase/Firestore initialization, calendar event store, and planning-target settings store.
- **Backend Service (`server/`)**: NestJS controllers, services, DTOs, and Prisma schemas for team/backend APIs.
- **Colocated Tests**: Tests live alongside the code they verify.

---

## 🎨 Design System & Theme Tokens

ViDo uses Tailwind CSS v4 with semantic CSS custom properties:

| Token | CSS Variable | Purpose |
|---|---|---|
| Canvas | `var(--canvas)` | Main application background. |
| Surface | `var(--surface)` | Card, panel, and modal containers. |
| Ink | `var(--ink)` | Primary headings and text. |
| Muted | `var(--muted)` | Secondary labels and metadata. |
| Faint | `var(--faint)` | Placeholder/disabled states. |
| Line | `var(--line)` | Borders and dividers. |
| Accent | `var(--accent)` | Primary brand/active states. |
| Accent Soft | `var(--accent-soft)` | Subdued tints and badges. |

> Explore accent palettes interactively at `/style-lab`.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend Framework** | [React 19](https://react.dev/) | Declarative, component-based user interface. |
| **Backend Framework** | [NestJS 11](https://nestjs.com/) | Dedicated TypeScript backend service (`server/`). |
| **ORM & Database** | [Prisma 6](https://www.prisma.io/) + [SQLite](https://www.sqlite.org/) | Local database (`dev.db`) with PostgreSQL production compatibility. |
| **Real-Time Push** | Server-Sent Events (SSE) | Push delivery for team invites and notification/task updates. |
| **Auth & Data** | [Firebase 13](https://firebase.google.com/) | Google sign-in and Firestore-backed planning data (local emulator by default). |
| **Bundler & Tooling** | [Vite 6](https://vite.dev/) | Fast HMR dev server and optimized builds. |
| **Language** | [TypeScript 5.7](https://www.typescriptlang.org/) | Strict compile-time safety and self-documenting models. |
| **Styling** | [Tailwind CSS v4](https://tailwindcss.com/) | Utility-first styling with semantic CSS variables. |
| **Routing** | [React Router 7](https://reactrouter.com/) | Declarative client-side routing with nested layouts. |
| **Icons** | [Lucide React](https://lucide.dev/) | Clean, accessible iconography. |
| **Testing** | [Vitest](https://vitest.dev/) + [Testing Library](https://testing-library.com/) | Rapid DOM/unit/integration tests in Happy DOM. |

---

## 🤝 Contributing Guidelines

Please review **[CONTRIBUTING.md](./CONTRIBUTING.md)** for the full workflow and PR checklist.

1. **Bug fixes**: search existing issues or open one with reproduction steps.
2. **UI redesigns/features**: start a discussion first to align on scope and design.
3. **Branch, implement, and verify**:
   ```bash
   npm run test:python && npm run eval:python && npm run typecheck && npm test && npm run build
   ```
4. **Open a PR** with issue/discussion links and before/after evidence for UI changes.

---

## 💡 Core Principles

```text
Clarity over complexity          History over guesswork
Privacy by default               Collaboration without noise
Trustworthy foundations          AI that drives useful action
```

---

<div align="center">

**ViDo** · Progress made visible  
Built with precision for focused execution.

</div>
