/**
 * ViDo's built-in knowledge base.
 *
 * These compact, human-written articles ship with the app, so answers are
 * instant and work offline or when the user has not uploaded any data. They
 * are indexed as regular RAG documents, which lets Graph RAG connect a user's
 * rows to general knowledge through shared topics and tags.
 */

import type { RAGDoc } from './engine';

export interface KnowledgeEntry {
  id: string;
  title: string;
  keys: string[];
  text: string;
  topic: string;
}

export const KNOWLEDGE_ENTRIES: KnowledgeEntry[] = [
  // ---------------------------------------------------------------- work
  {
    id: 'kb-prioritisation',
    title: 'Prioritisation',
    keys: ['prioritise', 'prioritize', 'priority', 'important', 'urgent', 'focus', 'order'],
    topic: 'planning',
    text: 'Prioritisation means deciding which work deserves attention first, based on impact and urgency rather than on what is loudest. A useful habit is to separate tasks that are truly important from those that are merely time-sensitive. When everything looks urgent, pick the one task that would make the biggest difference if it were finished today.',
  },
  {
    id: 'kb-deadlines',
    title: 'Deadlines and scheduling',
    keys: ['deadline', 'due', 'schedule', 'timeline', 'time', 'calendar', 'planning'],
    topic: 'planning',
    text: 'A deadline is the point by which a piece of work needs to be finished. Good scheduling works backwards from that date, leaving room for review and the unexpected. If a task has both a date and a time, treat the time as the hard stop and plan to finish comfortably before it.',
  },
  {
    id: 'kb-follow-ups',
    title: 'Follow-ups',
    keys: ['follow up', 'followup', 'reminder', 'check in', 'chase', 'waiting'],
    topic: 'planning',
    text: 'A follow-up is a short check to make sure something is still moving. It is most useful when it names the original promise, the current status and the next action, so the other person can reply quickly. Regular follow-ups prevent small delays from becoming missed deadlines.',
  },
  {
    id: 'kb-meetings',
    title: 'Meetings',
    keys: ['meeting', 'sync', 'standup', 'stand up', 'agenda', 'call', 'discussion'],
    topic: 'collaboration',
    text: 'A meeting is a deliberate conversation to share information, make decisions or agree on next steps. A short agenda and a clear owner for each decision usually make it far more productive. The meeting is only useful when it ends with named actions and dates.',
  },
  {
    id: 'kb-project-planning',
    title: 'Project planning',
    keys: ['project', 'plan', 'milestone', 'roadmap', 'scope', 'deliverable'],
    topic: 'planning',
    text: 'Project planning turns a broad goal into milestones, owners and dates. It works best when the final outcome is described first and the smaller steps are then arranged around it. A plan is a living document: revisit it when reality changes instead of treating it as fixed.',
  },
  {
    id: 'kb-delegation',
    title: 'Delegation',
    keys: ['delegate', 'assign', 'owner', 'responsibility', 'hand off', 'handoff'],
    topic: 'collaboration',
    text: 'Delegation is handing over both the work and the authority to make decisions about it. Clear delegation states the desired result, the deadline and what the person can decide on their own. It develops other people and frees the original owner to focus on higher-value work.',
  },
  {
    id: 'kb-notes',
    title: 'Notes and documentation',
    keys: ['note', 'notes', 'documentation', 'record', 'checkpoint', 'update', 'log'],
    topic: 'planning',
    text: 'Notes are a lightweight record of what was decided, what changed and what still needs doing. Short, dated entries are more useful than long prose because they show the history of a task at a glance. A good checkpoint captures the current state and the very next step.',
  },
  {
    id: 'kb-deep-work',
    title: 'Deep work and focus',
    keys: ['deep work', 'focus', 'concentration', 'distraction', 'attention', 'quiet'],
    topic: 'planning',
    text: 'Deep work is uninterrupted time spent on a demanding task that needs real thought. It requires removing notifications and protecting a block of time long enough to get past the initial friction. Even an hour a day of genuine focus usually outperforms a whole day of scattered activity.',
  },
  {
    id: 'kb-checklists',
    title: 'Checklists',
    keys: ['checklist', 'check list', 'verify', 'routine', 'steps', 'prepare'],
    topic: 'planning',
    text: 'A checklist is a short list of the steps that must not be forgotten. It is most valuable for routine work where mistakes come from rushing rather than from lack of knowledge. Keep a checklist brief, test it in real use, and update it whenever something goes wrong.',
  },
  {
    id: 'kb-retrospectives',
    title: 'Retrospectives',
    keys: ['retrospective', 'review', 'lesson', 'improve', 'reflect', 'learned'],
    topic: 'planning',
    text: 'A retrospective is a structured look back at what worked, what did not and what to change next time. It keeps the focus on the process rather than on blame. The most useful retrospectives finish with one or two concrete improvements that someone owns.',
  },
  {
    id: 'kb-estimation',
    title: 'Estimation',
    keys: ['estimate', 'effort', 'sizing', 'timebox', 'how long', 'guess'],
    topic: 'planning',
    text: 'An estimate is a reasoned guess about how long work will take, not a promise. It becomes more accurate when similar past work is used as a reference and when uncertainty is stated openly. Ranges are more honest than single numbers, and they can be narrowed as more is learned.',
  },
  {
    id: 'kb-risk',
    title: 'Risks and blockers',
    keys: ['risk', 'issue', 'blocker', 'mitigation', 'problem', 'dependency'],
    topic: 'planning',
    text: 'A risk is something that might go wrong; a blocker is something already stopping progress. Both should be written down with an owner and a plan, because hidden problems tend to surface late. The earlier a blocker is raised, the more options there are for dealing with it.',
  },
  {
    id: 'kb-status',
    title: 'Status reporting',
    keys: ['status', 'report', 'update', 'progress', 'summary', 'overview'],
    topic: 'collaboration',
    text: 'A good status report says where things stand, what changed since the last update and what happens next. It is short, factual and honest about anything that is slipping. People reading it should be able to tell immediately whether they need to act.',
  },
  {
    id: 'kb-decisions',
    title: 'Decision making',
    keys: ['decision', 'choose', 'option', 'tradeoff', 'trade off', 'evaluate'],
    topic: 'planning',
    text: 'Decisions improve when the options, the criteria and the trade-offs are written down before choosing. Reversible decisions can be made quickly; irreversible ones deserve more evidence and a wider group. Recording why a decision was made prevents the same debate from happening twice.',
  },
  {
    id: 'kb-time-management',
    title: 'Time management',
    keys: ['time', 'schedule', 'calendar', 'block', 'hours', 'plan day'],
    topic: 'planning',
    text: 'Time management is the practice of deciding in advance where your attention will go. Time blocking, where each important activity gets a slot in the calendar, is a simple way to make that decision concrete. Leaving some empty space is part of the plan, not a failure of it.',
  },

  // --------------------------------------------------------------- books
  {
    id: 'kb-gtd',
    title: 'Getting Things Done (David Allen)',
    keys: ['getting things done', 'gtd', 'david allen', 'inbox', 'capture', 'next action'],
    topic: 'books',
    text: 'Getting Things Done argues that the mind is for having ideas, not for holding them. Everything is captured in a trusted system, clarified into a concrete next action, and then organised by context and priority. The weekly review is what keeps the system trustworthy.',
  },
  {
    id: 'kb-deep-work-book',
    title: 'Deep Work (Cal Newport)',
    keys: ['deep work', 'cal newport', 'attention', 'shallow work', 'book'],
    topic: 'books',
    text: 'Deep Work makes the case that the ability to focus without distraction is becoming both rarer and more valuable. Newport suggests scheduling long, deliberate blocks of concentration and treating shallow tasks as secondary. The central idea is that better results come from sustained attention, not from more hours online.',
  },
  {
    id: 'kb-atomic-habits',
    title: 'Atomic Habits (James Clear)',
    keys: ['atomic habits', 'james clear', 'habit', 'cue', 'routine', 'reward'],
    topic: 'books',
    text: 'Atomic Habits focuses on tiny, repeatable changes rather than dramatic overhauls. Clear explains that habits run on a cue, a craving, a response and a reward, and that changing the environment is often easier than relying on willpower. Small improvements compound into large differences over time.',
  },
  {
    id: 'kb-checklist-manifesto',
    title: 'The Checklist Manifesto (Atul Gawande)',
    keys: ['checklist manifesto', 'atul gawande', 'checklist', 'error', 'surgery'],
    topic: 'books',
    text: 'The Checklist Manifesto shows how simple checklists dramatically reduce errors in complex, expert work. Gawande argues that failures often come from lapses in routine rather than from ignorance. A good checklist is short, precise and used at the moment the work happens.',
  },
  {
    id: 'kb-essentialism',
    title: 'Essentialism (Greg McKeown)',
    keys: ['essentialism', 'greg mckeown', 'less but better', 'say no', 'essential'],
    topic: 'books',
    text: 'Essentialism is the discipline of identifying what truly matters and saying no to almost everything else. McKeown rejects the idea that we can do it all and recommends protecting time for the few activities with the highest contribution. The result is fewer commitments, executed better.',
  },
  {
    id: 'kb-one-thing',
    title: 'The One Thing (Gary Keller)',
    keys: ['one thing', 'gary keller', 'domino', 'priority', 'focus'],
    topic: 'books',
    text: 'The One Thing argues that extraordinary results come from concentrating on a single most important priority. Keller uses the domino metaphor: one well-chosen action makes the next ones easier. The practical habit is to ask what one thing, done now, would make everything else easier or unnecessary.',
  },
  {
    id: 'kb-drive',
    title: 'Drive (Daniel Pink)',
    keys: ['drive', 'daniel pink', 'motivation', 'autonomy', 'mastery', 'purpose'],
    topic: 'books',
    text: 'Drive explains that lasting motivation comes less from rewards and punishment and more from autonomy, mastery and purpose. Pink shows that people do their best work when they have some control over what they do, can see themselves improving, and understand why the work matters.',
  },
  {
    id: 'kb-lean-startup',
    title: 'The Lean Startup (Eric Ries)',
    keys: ['lean startup', 'eric ries', 'experiment', 'mvp', 'validate', 'feedback'],
    topic: 'books',
    text: 'The Lean Startup treats new work as a series of experiments rather than a single big launch. Ries recommends building the smallest useful version, measuring real behaviour and learning quickly from the result. The build-measure-learn loop keeps effort focused on what customers actually need.',
  },
  {
    id: 'kb-thinking-fast-slow',
    title: 'Thinking, Fast and Slow (Daniel Kahneman)',
    keys: ['thinking fast and slow', 'kahneman', 'bias', 'system 1', 'system 2', 'decision'],
    topic: 'books',
    text: 'Thinking, Fast and Slow describes two modes of thought: a fast, intuitive one and a slow, deliberate one. Kahneman shows how the fast mode relies on shortcuts that can produce predictable biases. Slowing down for important decisions is the practical defence against those errors.',
  },
  {
    id: 'kb-dont-make-me-think',
    title: "Don't Make Me Think (Steve Krug)",
    keys: ["don't make me think", 'steve krug', 'usability', 'design', 'clarity'],
    topic: 'books',
    text: "Don't Make Me Think argues that good design should be obvious without effort. Krug recommends clear labels, conventional patterns and removing unnecessary choices so people can move forward confidently. Usability improves most when the next action is visible at a glance.",
  },

  // ------------------------------------------------------- personal life
  {
    id: 'kb-boating',
    title: 'Boating',
    keys: ['boat', 'boating', 'sail', 'kayak', 'cruise', 'rowing', 'water'],
    topic: 'personal',
    text: 'Boating is spending time on the water in a boat, whether that means sailing, rowing or using a small motorboat. A well-run trip includes checking the weather, planning a route, carrying safety gear such as life jackets, and making sure the boat has enough fuel or the right equipment. The main goal is usually to enjoy the water while getting everyone back safely.',
  },
  {
    id: 'kb-hiking',
    title: 'Hiking',
    keys: ['hike', 'hiking', 'trail', 'trek', 'walk', 'outdoors'],
    topic: 'personal',
    text: 'Hiking is a walk outdoors, usually on a trail, where the route and the conditions matter as much as the destination. Good preparation means choosing a route that matches the group, checking the weather, and carrying water, suitable footwear and a map. Starting early often makes the experience safer and more enjoyable.',
  },
  {
    id: 'kb-cooking',
    title: 'Cooking and meal planning',
    keys: ['cook', 'cooking', 'bake', 'meal', 'recipe', 'dinner', 'grocer', 'ingredient'],
    topic: 'personal',
    text: 'Cooking turns ingredients into a meal through a planned sequence of steps. Meal planning works backwards from what needs to be eaten, checks which ingredients are available, and lists only what is missing. Timing matters: good cooks start the slowest parts first and keep the final assembly simple.',
  },
  {
    id: 'kb-travel',
    title: 'Travel planning',
    keys: ['travel', 'trip', 'flight', 'hotel', 'vacation', 'holiday', 'booking'],
    topic: 'personal',
    text: 'Travel planning is the process of arranging transport, accommodation and a realistic itinerary before leaving. The most important details are dates, confirmation numbers, and the time needed to get between places. Building in a little slack prevents one delayed connection from ruining the whole trip.',
  },
  {
    id: 'kb-family-visit',
    title: 'Visiting family',
    keys: ['visit parents', 'family', 'parents', 'relatives', 'grandparents', 'visit'],
    topic: 'personal',
    text: 'Visiting family is a personal commitment that is easy to postpone until it becomes urgent. It usually involves agreeing on a time, arranging travel and deciding how long to stay. Putting the date in a calendar and setting a reminder a few days before is often all it takes to make it happen.',
  },
  {
    id: 'kb-cleaning',
    title: 'Cleaning and tidying',
    keys: ['clean', 'cleaning', 'tidy', 'laundry', 'wash', 'housework', 'declutter'],
    topic: 'personal',
    text: 'Cleaning restores a space or item to a usable, tidy state. It is much easier to maintain with a short regular routine than with a rare, large session. Working from the top down and finishing one area before starting another keeps the effort visible and avoids doing the same job twice.',
  },
  {
    id: 'kb-budgeting',
    title: 'Budgeting and bills',
    keys: ['budget', 'money', 'invoice', 'bill', 'pay', 'tax', 'finance', 'expense'],
    topic: 'personal',
    text: 'Budgeting is deciding in advance how money will be spent and then checking reality against that plan. The essentials are knowing what is due, when it is due, and how much is available. Recording each payment keeps records accurate and makes the next budget more realistic.',
  },
  {
    id: 'kb-exercise',
    title: 'Exercise',
    keys: ['exercise', 'workout', 'gym', 'run', 'fitness', 'training', 'sport'],
    topic: 'personal',
    text: 'Exercise is planned physical activity that keeps the body strong and healthy. The best routine is one that can be repeated: a modest session done regularly beats an ambitious plan abandoned after a week. Recovery days matter too, because the body improves between sessions.',
  },
  {
    id: 'kb-health',
    title: 'Health appointments',
    keys: ['doctor', 'appointment', 'health', 'checkup', 'medical', 'dentist', 'clinic'],
    topic: 'personal',
    text: 'A health appointment is a planned visit to a professional for advice, a check-up or treatment. Useful preparation includes bringing a short list of symptoms or questions and any relevant records. Writing down the advice given makes it much easier to follow afterwards.',
  },
  {
    id: 'kb-gardening',
    title: 'Gardening',
    keys: ['garden', 'gardening', 'plant', 'lawn', 'flower', 'seeds'],
    topic: 'personal',
    text: 'Gardening is the practice of growing plants or maintaining an outdoor space. Success depends on matching plants to the season and the light available, then keeping up with watering and weeding. Small, regular attention is more effective than occasional bursts of hard work.',
  },
  {
    id: 'kb-pets',
    title: 'Pet care',
    keys: ['pet', 'dog', 'cat', 'vet', 'feed', 'walking the dog', 'animal'],
    topic: 'personal',
    text: 'Pet care is a daily responsibility that includes feeding, exercise, cleanliness and regular veterinary check-ups. Animals thrive on routine, so fixed times for walks and meals reduce stress for everyone. Preventive care is usually easier and cheaper than treating a problem later.',
  },
  {
    id: 'kb-shopping',
    title: 'Shopping',
    keys: ['shop', 'shopping', 'buy', 'purchase', 'store', 'order'],
    topic: 'personal',
    text: 'Shopping is the act of obtaining something that is needed or wanted. A short list and a clear budget stop a quick errand from turning into an expensive one. For anything important, comparing price, quality and return policy before buying saves regret later.',
  },
  {
    id: 'kb-reading',
    title: 'Reading and learning',
    keys: ['read', 'reading', 'book', 'library', 'study', 'learn'],
    topic: 'personal',
    text: 'Reading is one of the most reliable ways to learn something new or see a familiar problem differently. Setting a small, specific goal, such as a few pages a day, is more sustainable than occasional long sessions. Summarising what was read in your own words is the best test of whether it was understood.',
  },
  {
    id: 'kb-language-learning',
    title: 'Learning a language',
    keys: ['language', 'spanish', 'french', 'german', 'vocabulary', 'practice', 'speak'],
    topic: 'personal',
    text: 'Learning a language is built on frequent, manageable practice rather than rare cramming. Short daily sessions that combine vocabulary, listening and speaking produce faster progress than long grammar studies alone. Making mistakes in real conversations is part of the learning process.',
  },
  {
    id: 'kb-events',
    title: 'Event planning',
    keys: ['event', 'party', 'birthday', 'wedding', 'celebration', 'gathering'],
    topic: 'personal',
    text: 'Event planning means deciding who, when, where and what, then working backwards from the date. The first things to secure are the venue and the people who need advance notice. A short checklist for food, supplies and setup prevents the common last-minute scramble.',
  },

  // ---------------------------------------------------------------- data
  {
    id: 'kb-confidence',
    title: 'Prediction confidence',
    keys: ['confidence', 'probability', 'likely', 'prediction', 'chance', 'score', 'percent'],
    topic: 'data',
    text: 'A prediction confidence or probability is the model\'s estimate of how likely an outcome is, based on patterns in the data it learned from. A high number means the observed features resemble past examples of that outcome, not that the result is guaranteed. Treat it as a guide for where to look first, and check the underlying records before acting.',
  },
  {
    id: 'kb-split',
    title: 'Train and test split',
    keys: ['train', 'test', 'split', 'validation', 'holdout', 'generalise', 'generalize'],
    topic: 'data',
    text: 'A train and test split keeps part of the data aside so a model can be checked on examples it has never seen. Training on everything and then testing on the same rows would give a flattering but misleading score. The gap between training and test performance shows whether the model really learned a pattern.',
  },
  {
    id: 'kb-accuracy',
    title: 'Accuracy',
    keys: ['accuracy', 'correct', 'performance', 'metric', 'score', 'right'],
    topic: 'data',
    text: 'Accuracy is the share of predictions that were correct. It is easy to understand but can be misleading when one outcome is much more common than another. Looking at precision, recall and the counts behind the percentage gives a fuller picture.',
  },
  {
    id: 'kb-precision-recall',
    title: 'Precision and recall',
    keys: ['precision', 'recall', 'false positive', 'false negative', 'sensitivity'],
    topic: 'data',
    text: 'Precision asks how many of the predicted positives were actually positive, while recall asks how many of the real positives were found. Raising one often lowers the other, so the right balance depends on which mistake costs more. Reading both together is much more informative than a single accuracy number.',
  },
  {
    id: 'kb-overfitting',
    title: 'Overfitting',
    keys: ['overfit', 'overfitting', 'memorise', 'memorize', 'generalise', 'generalize', 'too complex'],
    topic: 'data',
    text: 'Overfitting happens when a model memorises the training rows instead of learning a pattern that transfers to new data. The tell-tale sign is excellent training performance but noticeably worse test performance. Simpler models, more data and a proper test split are the usual remedies.',
  },
  {
    id: 'kb-features',
    title: 'Features and columns',
    keys: ['feature', 'column', 'importance', 'predictor', 'field', 'variable'],
    topic: 'data',
    text: 'Features are the columns a model uses to make a prediction. Useful features carry information about the outcome without leaking the answer itself. Comparing features by how much they improve test performance is a practical way to judge which ones really matter.',
  },
  {
    id: 'kb-categories',
    title: 'Categories and labels',
    keys: ['category', 'categories', 'class', 'label', 'group', 'type'],
    topic: 'data',
    text: 'A category is a named group that rows can belong to, and a label is the outcome being predicted. Clear, consistent category names make both analysis and answers much easier to follow. When categories overlap or change spelling, grouping them first usually improves the results.',
  },
  {
    id: 'kb-trends',
    title: 'Trends and patterns',
    keys: ['trend', 'pattern', 'over time', 'history', 'increase', 'decrease'],
    topic: 'data',
    text: 'A trend is a direction that persists across several records or over time. A single unusual row is not a trend, so it is worth checking how many points support the pattern. Looking at the time order, not just the average, is what turns a list of values into a useful insight.',
  },
  {
    id: 'kb-outliers',
    title: 'Outliers',
    keys: ['outlier', 'anomaly', 'unusual', 'extreme', 'strange', 'exception'],
    topic: 'data',
    text: 'An outlier is a record that sits far away from the rest of the data. It can be a genuine and important case, or simply a typing error, so it should be inspected rather than deleted automatically. Deciding what to do with it often changes the conclusions more than expected.',
  },
  {
    id: 'kb-missing-data',
    title: 'Missing data',
    keys: ['missing', 'blank', 'null', 'incomplete', 'empty', 'unknown'],
    topic: 'data',
    text: 'Missing data means a value was not recorded, and the reason it is missing can matter. Filling gaps with guesses can hide real problems, so it is better to understand the cause first. In a small dataset, a few blanks can noticeably change an average or a prediction.',
  },

  // -------------------------------------------------------------- general
  {
    id: 'kb-getting-started',
    title: 'Getting started with ViDo',
    keys: ['start', 'begin', 'help', 'how do i', 'how to', 'use', 'upload', 'question'],
    topic: 'general',
    text: 'ViDo learns from a CSV or Excel file that you upload to ML Lab, trains automatically, and then answers questions about those rows. You can ask about a single record, compare groups, or request a plain-language explanation of a topic. If you have not uploaded anything yet, you can still ask about general subjects and ViDo will answer from its built-in knowledge.',
  },
];

/** Stable document array so RAG indexes are built once. */
export const KNOWLEDGE_DOCS: RAGDoc[] = KNOWLEDGE_ENTRIES.map((entry) => ({
  id: entry.id,
  title: entry.title,
  text: `${entry.title}. ${entry.text}`,
  tags: `topic=${entry.topic}, knowledge=true, ${entry.keys.map((key) => `key=${key}`).join(', ')}`,
  priority: '',
  assignee: '',
  edges: [],
}));

const words = (value: string): string[] =>
  String(value || '')
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9'_-]*/g) || [];

/** Find the article that best matches a subject phrase. */
export function findKnowledge(subject: string): KnowledgeEntry | null {
  const text = words(subject).join(' ');
  if (!text.trim()) return null;

  let best: KnowledgeEntry | null = null;
  let bestScore = 0;

  for (const entry of KNOWLEDGE_ENTRIES) {
    let score = 0;
    for (const key of entry.keys) {
      const phrase = words(key).join(' ');
      if (!phrase) continue;
      if (text.includes(phrase)) score += phrase.split(' ').length * 3;
    }
    if (words(entry.title).some((word) => word.length > 3 && text.includes(word))) score += 2;
    if (score > bestScore) {
      bestScore = score;
      best = entry;
    }
  }

  return bestScore > 0 ? best : null;
}
