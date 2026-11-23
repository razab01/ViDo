/**
 * Lightweight, offline word understanding for ViDo.
 *
 * This is not a language model. It is a compact dictionary of word meanings
 * that lets ViDo explain a *combination* of words ("financial report for this
 * quarter") instead of matching one keyword like "report". Everything here is
 * plain text and runs in microseconds, so answers stay instant and offline.
 */

type SenseKind = 'action' | 'thing' | 'quality' | 'person' | 'time' | 'place' | 'event';

interface Sense {
  gloss: string;
  kind: SenseKind;
}

const a = (gloss: string): Sense => ({ gloss, kind: 'action' });
const n = (gloss: string): Sense => ({ gloss, kind: 'thing' });
const q = (gloss: string): Sense => ({ gloss, kind: 'quality' });
const p = (gloss: string): Sense => ({ gloss, kind: 'person' });
const t = (gloss: string): Sense => ({ gloss, kind: 'time' });
const l = (gloss: string): Sense => ({ gloss, kind: 'place' });
const e = (gloss: string): Sense => ({ gloss, kind: 'event' });

/** Common English words used in tasks, notes and everyday topics. */
const WORD_SENSES: Record<string, Sense> = {
  // actions
  analyse: a('looking at the details carefully'), analyze: a('looking at the details carefully'),
  approve: a('giving formal permission to go ahead'), audit: a('checking the records for accuracy'),
  build: a('creating something new'), calculate: a('working out the numbers'),
  call: a('having a short phone or video conversation'), celebrate: a('marking a happy occasion'),
  check: a('making sure something is correct'), clean: a('tidying and washing a space'),
  cook: a('preparing food'), create: a('making something new'),
  deliver: a('handing over the finished work'), deploy: a('putting a new version into use'),
  design: a('working out how something should look and work'),
  edit: a('improving and correcting a piece of writing'),
  exercise: a('doing planned physical activity'), fix: a('repairing something that is broken'),
  follow: a('checking that something is still moving forward'),
  garden: a('looking after plants and an outdoor space'), hike: a('taking a long walk outdoors'),
  launch: a('releasing something for people to use'), learn: a('practising until something becomes familiar'),
  move: a('changing home or location'), organize: a('putting things into a clear order'),
  organise: a('putting things into a clear order'), paint: a('covering a surface with colour'),
  pay: a('giving money for something'),
  prepare: a('getting everything ready'), present: a('showing information to an audience'),
  read: a('looking at written words to learn something'), release: a('making a new version available'),
  remind: a('prompting someone about something they must do'), repair: a('putting something back into working order'),
  research: a('finding out facts about a subject'), review: a('checking something carefully again'),
  run: a('moving quickly on foot, or operating something'), sail: a('travelling on water using wind'),
  schedule: a('choosing the time for something'), send: a('getting a message or item to someone'),
  study: a('spending time learning a subject'), submit: a('handing in finished work'),
  swim: a('moving through water'), test: a('trying something to see if it works'),
  train: a('practising or teaching a skill'), travel: a('going from one place to another'),
  verify: a('confirming that something is true'), visit: a('spending time with someone or somewhere'),
  write: a('producing a piece of text'), boat: a('travelling on water in a small vessel'),
  boating: a('spending time on the water in a boat'), fishing: a('trying to catch fish'),
  sailing: a('travelling on water using wind'), cooking: a('preparing food'),
  shopping: a('buying the things that are needed'), cleaning: a('tidying and washing a space'),
  hiking: a('taking a long walk outdoors'), swimming: a('moving through water'),
  gardening: a('looking after plants and an outdoor space'), running: a('moving quickly on foot'),

  // things
  app: n('a program used on a phone or computer'), applicant: p('someone applying for a role'),
  assignment: n('a piece of work given to someone'), backup: n('a second copy kept for safety'),
  bill: n('a request for payment'), blog: n('a short article published online'),
  book: n('a written work you read'), brief: n('a short summary of what is needed'),
  budget: n('a plan for expected income and spending'), bug: n('a fault in a program'),
  campaign: n('a planned effort to reach a goal'), checklist: n('a short list of steps not to forget'),
  code: n('the instructions that tell a program what to do'), content: n('the written or visual material in something'),
  contract: n('a written agreement between parties'), data: n('the facts and figures collected for analysis'),
  database: n('an organised store of information'), deadline: n('the final time by which work must be finished'),
  delivery: n('bringing an item to its destination'), dependency: n('something that must happen first'),
  document: n('a written record of information'), draft: n('a first version of a piece of work'),
  email: n('a written message sent online'), estimate: n('a reasoned guess about cost or time'),
  expense: n('money spent on something'), feature: n('a distinct part or capability of a product'),
  fee: n('a charge for a service'), file: n('a stored piece of information'),
  forecast: n('a prediction about what will happen'), form: n('a structured sheet of questions to answer'),
  gear: n('the equipment needed for an activity'), goal: n('the result someone is trying to reach'),
  gift: n('something given to someone as a present'), invoice: n('a written request for payment'),
  issue: n('a problem that needs attention'), itinerary: n('the planned route and timetable of a trip'),
  keys: n('the objects used to open locks'), license: n('official permission to use something'),
  luggage: n('the bags taken on a trip'), meal: n('a planned occasion for eating'),
  medicine: n('something taken to treat an illness'), message: n('a short piece of communication'),
  milestone: n('an important point of progress in a plan'), model: n('a system that learns patterns from data'),
  note: n('a short written record'), objective: n('a specific result to achieve'),
  order: n('a request to supply something'), password: n('a secret word used to access an account'),
  payment: n('money handed over for something'), passport: n('an official travel document'),
  plan: n('a description of the steps and timing'), post: n('a short piece published online'),
  presentation: n('a talk that shows information to an audience'), price: n('the amount charged for something'),
  priority: n('the thing that deserves attention first'), product: n('something made or offered to customers'),
  proposal: n('a written plan put forward for consideration'), quote: n('a stated price or short quoted passage'),
  recipe: n('the instructions for making a dish'), refund: n('money given back to a customer'),
  rent: n('money paid to use a home or item'), report: n('a clear summary of what happened'),
  roadmap: n('a high-level plan of future work'), risk: n('something that might go wrong'),
  server: n('a computer that provides a service to others'), shipment: n('goods sent to a destination'),
  spreadsheet: n('a grid of rows and columns for data'), stock: n('the goods available to sell'),
  strategy: n('the overall approach to reaching a goal'), subscription: n('a recurring payment for access'),
  supplier: n('a person or company that provides goods'), survey: n('a set of questions asked to collect opinions'),
  tax: n('money paid to the government'), ticket: n('a document that allows entry or travel'),
  visa: n('official permission to enter a country'), website: n('a collection of pages on the internet'),

  // qualities
  active: q('currently being worked on'), annual: q('happening once a year'), archived: q('kept but no longer active'),
  big: q('large in size or importance'), clear: q('easy to understand'), closed: q('no longer open or active'),
  complex: q('made of many connected parts'), creative: q('using imagination to make something new'),
  daily: q('happening every day'), detailed: q('including many specific facts'), difficult: q('hard to do'),
  done: q('finished'), easy: q('not hard to do'), external: q('outside the immediate team or group'),
  final: q('the last and settled version'), financial: q('money, budgets and accounts'),
  first: q('coming before the others'), high: q('above the usual level'), home: q('relating to where someone lives'),
  important: q('having real value or impact'), internal: q('inside the immediate team or group'),
  large: q('big in size or amount'), last: q('coming after the others'), legal: q('relating to the law'),
  low: q('below the usual level'), medium: q('in the middle between low and high'),
  medical: q('relating to health and treatment'), monthly: q('happening once a month'),
  new: q('recently made or started'), next: q('coming immediately after this one'),
  old: q('made or started a long time ago'), open: q('still waiting to be done'),
  overdue: q('past the agreed deadline'), pending: q('waiting to be decided or done'),
  personal: q('relating to someone’s own life'), private: q('not shared with others'),
  public: q('available to everyone'), quick: q('taking very little time'), rough: q('not yet polished or exact'),
  simple: q('easy to understand or do'), slow: q('taking more time than expected'),
  small: q('little in size or amount'), technical: q('relating to how something works'),
  upcoming: q('happening soon'), urgent: q('needing attention immediately'),
  weekly: q('happening once a week'), work: q('relating to someone’s job or duties'),
  quarterly: q('relating to a three-month period'),

  // people
  boss: p('the person in charge'), brother: p('a male sibling'), child: p('a young person'),
  colleague: p('someone working in the same place'), customer: p('someone who buys from a business'),
  dad: p('a father'), daughter: p('a female child'), dentist: p('a professional who looks after teeth'),
  doctor: p('a professional who looks after health'), family: p('parents, children and close relatives'),
  father: p('a male parent'), friend: p('someone you know well and like'),
  grandparent: p('a parent of a parent'), guest: p('someone invited to an event'),
  host: p('someone who welcomes guests'), husband: p('a married man'), kid: p('a child'),
  manager: p('someone who organises a team'), mum: p('a mother'), mom: p('a mother'),
  mother: p('a female parent'), parent: p('a mother or father'), parents: p('a mother and father'),
  partner: p('someone who shares life or work'), patient: p('someone receiving medical care'),
  relative: p('a member of the family'), sister: p('a female sibling'), sibling: p('a brother or sister'),
  son: p('a male child'), spouse: p('a husband or wife'), student: p('someone who is learning'),
  teacher: p('someone who teaches'), teammate: p('someone on the same team'),
  technician: p('someone who repairs technical equipment'), vet: p('a professional who looks after animals'),
  wife: p('a married woman'),

  // time
  afternoon: t('the early part of the day after midday'),
  april: t('the fourth month'), august: t('the eighth month'), december: t('the twelfth month'),
  evening: t('the later part of the day'),
  february: t('the second month'), friday: t('the day before Saturday'),
  hour: t('a period of sixty minutes'), january: t('the first month'), july: t('the seventh month'),
  june: t('the sixth month'), march: t('the third month'), may: t('the fifth month'),
  monday: t('the first day of the working week'), month: t('a period of about thirty days'),
  morning: t('the early part of the day'), november: t('the eleventh month'),
  october: t('the tenth month'), saturday: t('the day after Friday'),
  september: t('the ninth month'), sunday: t('the day after Saturday'),
  thursday: t('the day before Friday'), today: t('the current day'),
  tomorrow: t('the day after today'), tuesday: t('the day after Monday'),
  wednesday: t('the day after Tuesday'), week: t('a period of seven days'),
  weekend: t('Saturday and Sunday'), year: t('a period of twelve months'),
  yesterday: t('the day before today'), minute: t('a period of sixty seconds'),
  night: t('the dark part of the day'), q1: t('the first three-month period of the year'),
  q2: t('the second three-month period of the year'), q3: t('the third three-month period of the year'),
  q4: t('the final three-month period of the year'), quarter: t('a three-month period'),

  // places
  airport: l('a place where people travel by plane'), apartment: l('a home in a shared building'),
  bank: l('a place that looks after money'), beach: l('the edge of a sea or lake'),
  clinic: l('a place for medical appointments'), garage: l('a place to keep a car or tools'),
  gym: l('a place for physical exercise'),
  hospital: l('a place for medical treatment'), house: l('a building where someone lives'),
  kitchen: l('the room where food is prepared'), lake: l('a large body of water surrounded by land'),
  office: l('a place where people work at desks'), park: l('a public outdoor space'),
  restaurant: l('a place where meals are served'), river: l('a flowing body of water'),
  room: l('a space inside a building'), school: l('a place where children learn'),
  sea: l('a large body of salt water'), shop: l('a place where things are sold'),
  station: l('a place where trains or buses stop'), store: l('a place where things are sold'),
  university: l('a place where adults study'),

  // events
  appointment: e('an agreed time to meet a professional'),
  checkup: e('a routine medical examination'), conference: e('a large meeting around a shared subject'),
  event: e('a planned occasion with people'), exam: e('a formal test of knowledge'),
  holiday: e('a period of time away from work'), interview: e('a formal conversation to assess a person'),
  match: e('a sports contest between teams'),
  meeting: e('a deliberate conversation to share information or decide something'),
  party: e('a social gathering for celebration'), rehearsal: e('a practice session before a performance'),
  retrospective: e('a structured look back at work'), sync: e('a short team update meeting'),
  trip: e('a journey to another place'), vacation: e('a period of time away from work'),
  wedding: e('a ceremony where two people marry'), workshop: e('a focused session to learn or plan together'),
  birthday: e('the day someone was born'), game: e('an activity with rules played for fun'),
  music: n('sounds arranged in a pleasing pattern'), movie: n('a story shown on a screen'),
  sport: n('a physical activity with rules'), bookclub: e('a group meeting to discuss a book'),
};

const CONNECTORS = new Set([
  'a', 'an', 'and', 'at', 'by', 'during', 'for', 'from', 'in', 'my', 'of', 'on', 'our',
  'the', 'their', 'this', 'that', 'these', 'those', 'to', 'with', 'your', 'about', 're',
]);

const normalize = (value: string): string[] =>
  String(value || '')
    .toLowerCase()
    .match(/[a-z0-9]+/g) || [];

/** Content words only, with connectors and one-letter noise removed. */
export const subjectKeywords = (subject: string): string[] =>
  Array.from(
    new Set(
      normalize(subject)
        .filter((word) => !CONNECTORS.has(word) && word.length > 1)
        .map((word) => word.replace(/(ies)$/, 'y').replace(/(ing|ed|es|s)$/, ''))
        .filter(Boolean)
    )
  );

const lookupSense = (word: string): Sense | undefined => {
  if (WORD_SENSES[word]) return WORD_SENSES[word];
  const stem = word.replace(/(ies)$/, 'y').replace(/(ing|ed|es|s)$/, '');
  return WORD_SENSES[word] || WORD_SENSES[stem];
};

const phrase = (values: string[]): string => {
  const clean = Array.from(new Set(values.map((v) => v.trim()).filter(Boolean)));
  if (clean.length <= 1) return clean[0] || '';
  if (clean.length === 2) return `${clean[0]} and ${clean[1]}`;
  return `${clean.slice(0, -1).join(', ')} and ${clean[clean.length - 1]}`;
};

const capitalize = (value: string): string =>
  value ? value.charAt(0).toUpperCase() + value.slice(1) : '';

/** Strip question wording so "What is quantum computing?" becomes a topic. */
export const topicPhrase = (question: string): string =>
  String(question || '')
    .trim()
    .replace(/\?+$/, '')
    .replace(/^(?:tell me about|explain|describe|summari[sz]e|what about|about)\s+/i, '')
    .replace(
      /^(?:what|which|who|when|where|why|how)\s+(?:is|are|was|were|do|does|did|can|could|should|would)\s+/i,
      ''
    )
    .replace(/^(?:is|are|what's|what is)\s+/i, '')
    .trim();

/**
 * Build a natural sentence that explains each meaningful word in a subject.
 * Returns '' when nothing is understood, so callers can fall back to another source.
 */
export function describeSubject(subject: string): string {
  const title = String(subject || '').replace(/\s+/g, ' ').trim();
  if (!title) return '';

  const words = normalize(title).filter((word) => !CONNECTORS.has(word) && word.length > 1);
  const senses = words.map(lookupSense).filter((sense): sense is Sense => Boolean(sense));
  if (!senses.length) return '';

  const mainIndex = (() => {
    for (let index = senses.length - 1; index >= 0; index -= 1) {
      if (senses[index].kind === 'action' || senses[index].kind === 'thing' || senses[index].kind === 'event')
        return index;
    }
    return senses.length - 1;
  })();
  const main = senses[mainIndex];
  const rest = senses.filter((_, index) => index !== mainIndex);
  const ofKind = (kind: SenseKind) => rest.filter((sense) => sense.kind === kind).map((sense) => sense.gloss);

  let core = main.gloss;
  const qualities = ofKind('quality');
  if (qualities.length) core += `, focused on ${phrase(qualities)}`;

  const extras = rest
    .filter((sense) => sense.kind !== 'quality' && sense.kind !== 'person' && sense.kind !== 'place' && sense.kind !== 'time')
    .map((sense) => sense.gloss);
  if (extras.length) core += `, including ${phrase(extras)}`;

  const people = ofKind('person');
  if (people.length) core += `${/ with /.test(main.gloss) ? ' alongside' : ' with'} ${phrase(people)}`;

  const places = ofKind('place');
  if (places.length) core += ` in ${phrase(places)}`;

  const times = ofKind('time');
  if (times.length) core += ` for ${phrase(times)}`;

  return `${capitalize(title)} is ${core}.`;
}
