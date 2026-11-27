import * as THREE from 'three';
import type { SceneKey } from './theme';

export interface SceneWorld {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  update: (time: number, delta: number) => void;
  resize: (width: number, height: number) => void;
  dispose: () => void;
}

type ColorStop = [number, string];

const rand = (min: number, max: number) => min + Math.random() * (max - min);

/** Stable placement noise for the rainforest, so a preview never reshuffles. */
function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function standard(color: string, roughness = 0.82, metalness = 0): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function gradientTexture(stops: ColorStop[]): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 4;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createLinearGradient(0, 0, 0, 256);
    stops.forEach(([offset, color]) => gradient.addColorStop(offset, color));
    context.fillStyle = gradient;
    context.fillRect(0, 0, 4, 256);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function glowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.28, 'rgba(255,255,255,0.7)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 64);
  }
  return new THREE.CanvasTexture(canvas);
}

/** Deformed icosahedron with a height-based ice/snow gradient. */
function organicGeometry(
  radius: number,
  detail: number,
  seed: number,
  lowColor: string,
  highColor: string,
  warp = 0.16,
): THREE.BufferGeometry {
  const geometry = new THREE.IcosahedronGeometry(radius, detail);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const low = new THREE.Color(lowColor);
  const high = new THREE.Color(highColor);
  const colors: number[] = [];

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const y = position.getY(index);
    const z = position.getZ(index);
    const variation =
      Math.sin(x * 3.1 + seed) * Math.cos(y * 2.7 - seed * 0.7) * Math.sin(z * 2.3 + seed * 0.4);
    const factor = 1 + variation * warp + (y > 0 ? 0.05 : -0.03);
    position.setXYZ(index, x * factor, y * factor, z * factor);
    const shade = THREE.MathUtils.clamp((y / radius + 1) * 0.5, 0, 1);
    const color = low.clone().lerp(high, shade);
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

/** Thin rod between two points, used for canopy ribs, rigging, and stems. */
function rodBetween(from: THREE.Vector3, to: THREE.Vector3, radius: number, color: string): THREE.Mesh {
  const direction = new THREE.Vector3().subVectors(to, from);
  const rod = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, direction.length(), 6),
    standard(color, 0.82),
  );
  rod.position.copy(from).addScaledVector(direction, 0.5);
  rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return rod;
}

/** Tapered, drooping palm leaf with a raised central vein. */
function frondGeometry(
  length: number,
  width: number,
  rootColor = '#2f6b49',
  tipColor = '#82b66f',
): THREE.BufferGeometry {
  const segments = 9;
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const root = new THREE.Color(rootColor);
  const tip = new THREE.Color(tipColor);

  for (let index = 0; index <= segments; index += 1) {
    const t = index / segments;
    const x = t * length;
    const y = Math.sin(t * Math.PI) * 0.2 - t * t * 0.66;
    const halfWidth = Math.sin(Math.PI * t) * width * (1 - t * 0.22);
    const color = root.clone().lerp(tip, t);
    positions.push(x, y, -halfWidth, x, y + 0.055 * Math.sin(t * Math.PI), 0, x, y, halfWidth);
    for (let vertex = 0; vertex < 3; vertex += 1) {
      colors.push(color.r, color.g, color.b);
    }
  }

  for (let index = 0; index < segments; index += 1) {
    const a = index * 3;
    const b = (index + 1) * 3;
    indices.push(a, a + 1, b, a + 1, b + 1, b);
    indices.push(a + 1, a + 2, b + 1, a + 2, b + 2, b + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Triangular sail with a soft billow so it does not read as a flat card. */
function sailGeometry(
  apex: THREE.Vector3,
  left: THREE.Vector3,
  right: THREE.Vector3,
  billow: number,
): THREE.BufferGeometry {
  const rows = 8;
  const positions: number[] = [];
  const indices: number[] = [];

  for (let row = 0; row <= rows; row += 1) {
    const t = row / rows;
    const rowLeft = apex.clone().lerp(left, t);
    const rowRight = apex.clone().lerp(right, t);
    for (let column = 0; column <= row; column += 1) {
      const u = row === 0 ? 0 : column / row;
      const point = rowLeft.clone().lerp(rowRight, u);
      point.z += Math.sin(Math.PI * t) * Math.sin(Math.PI * u) * billow;
      positions.push(point.x, point.y, point.z);
    }
  }

  const startOf = (row: number) => (row * (row + 1)) / 2;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      const a = startOf(row) + column;
      const b = a + 1;
      const c = startOf(row + 1) + column;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Fabric ribs and a scalloped hem for sun parasols. */
function addCanopyDetails(group: THREE.Group, radius: number, canopyY: number, color: string): void {
  const apex = canopyY + radius * 0.42;
  for (let index = 0; index < 8; index += 1) {
    const angle = (index / 8) * Math.PI * 2;
    group.add(
      rodBetween(
        new THREE.Vector3(0, apex - 0.03, 0),
        new THREE.Vector3(Math.cos(angle) * radius * 0.97, canopyY + 0.02, Math.sin(angle) * radius * 0.97),
        0.012,
        color,
      ),
    );
  }

  const points: THREE.Vector3[] = [];
  for (let step = 0; step <= 64; step += 1) {
    const angle = (step / 64) * Math.PI * 2;
    const scallop = 1 + Math.cos(angle * 8) * 0.035;
    points.push(
      new THREE.Vector3(
        Math.cos(angle) * radius * scallop,
        canopyY + Math.cos(angle * 8) * 0.012,
        Math.sin(angle) * radius * scallop,
      ),
    );
  }
  group.add(
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }),
    ),
  );
}

interface WorldContext {
  world: THREE.Scene;
  textures: THREE.Texture[];
  compact: boolean;
  glow: THREE.Texture;
  update: Array<(time: number, delta: number) => void>;
}

function addSky(context: WorldContext, stops: ColorStop[]): void {
  const texture = gradientTexture(stops);
  context.textures.push(texture);
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(120, 32, 16),
    new THREE.MeshBasicMaterial({
      map: texture,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
    }),
  );
  context.world.add(mesh);
}

/**
 * Procedural polar sky: a deep vertical gradient, a faint green horizon glow,
 * a soft diagonal milky band, and tiny stars that twinkle on the same
 * twelve-second cycle as the aurora. No textures, no generated scene images.
 */
function addAuroraSky(context: WorldContext): void {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uLoop: { value: 12 },
    },
    vertexShader: `
      varying vec3 vDirection;
      void main() {
        vDirection = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      uniform float uTime;
      uniform float uLoop;
      varying vec3 vDirection;

      void main() {
        vec3 d = normalize(vDirection);
        float y = max(d.y, 0.0);
        float t = 6.2831853 * uTime / uLoop;

        vec3 color = mix(vec3(0.012, 0.042, 0.078), vec3(0.002, 0.008, 0.026), pow(y, 0.45));
        color += vec3(0.0, 0.10, 0.06) *
          exp(-abs(d.y - 0.12) * 7.0) * (0.75 + 0.25 * sin(t));

        vec3 cell = d * 190.0;
        vec3 id = floor(cell);
        float h = fract(sin(dot(id, vec3(12.9, 78.2, 45.1))) * 43758.5);
        float star = smoothstep(0.3, 0.0, length(fract(cell) - 0.5)) *
          step(0.992, h) * (0.6 + 0.4 * sin(t * 2.0 + h * 80.0)) *
          smoothstep(-0.02, 0.15, d.y);
        color += vec3(0.9, 0.95, 1.0) * star * 1.6;

        float milky = pow(max(0.0, 1.0 - abs(dot(d, normalize(vec3(0.6, 0.3, -0.7))) * 3.0)), 2.0);
        color += vec3(0.05, 0.07, 0.1) * milky * 0.5;
        gl_FragColor = vec4(color, 1.0);
      }
    `,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
  });
  material.toneMapped = false;
  const sky = new THREE.Mesh(new THREE.SphereGeometry(120, 32, 16), material);
  sky.renderOrder = -1;
  context.world.add(sky);
  context.update.push((time) => {
    material.uniforms.uTime.value = time;
  });
}

function addLights(
  context: WorldContext,
  options: {
    sky: string;
    ground: string;
    sun: string;
    sunPower: number;
    ambient: number;
    sunPosition?: [number, number, number];
  },
): THREE.AmbientLight {
  const hemi = new THREE.HemisphereLight(options.sky, options.ground, 1.1);
  const sun = new THREE.DirectionalLight(options.sun, options.sunPower);
  const sunPosition: [number, number, number] = options.sunPosition ?? [-6, 11, 7];
  sun.position.set(...sunPosition);
  if (!context.compact) {
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -16;
    sun.shadow.camera.right = 16;
    sun.shadow.camera.top = 16;
    sun.shadow.camera.bottom = -16;
    sun.shadow.bias = -0.0008;
  }
  const ambient = new THREE.AmbientLight(options.sky, options.ambient);
  context.world.add(hemi, sun, ambient);
  return ambient;
}

function addStars(context: WorldContext, count: number, color = '#eaf4ff'): void {
  const positions = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const theta = rand(0, Math.PI * 2);
    const phi = rand(0.12, Math.PI * 0.46);
    const radius = 72;
    positions[index * 3] = Math.sin(phi) * Math.cos(theta) * radius;
    positions[index * 3 + 1] = Math.cos(phi) * radius;
    positions[index * 3 + 2] = Math.sin(phi) * Math.sin(theta) * radius;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color,
    size: 0.55,
    map: context.glow,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const stars = new THREE.Points(geometry, material);
  context.world.add(stars);
  context.update.push((time) => {
    (material as THREE.PointsMaterial).opacity = 0.72 + Math.sin(time * 0.8) * 0.16;
  });
}

/** Falling snow for the polar night, with a slow sideways drift and a soft reset. */
function addPolarSnowfall(context: WorldContext, count: number): void {
  const positions = new Float32Array(count * 3);
  const speeds = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    positions[index * 3] = rand(-38, 38);
    positions[index * 3 + 1] = rand(0, 34);
    positions[index * 3 + 2] = rand(-46, 20);
    speeds[index] = rand(1.4, 3.2);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: '#dff4ff',
    size: 0.22,
    map: context.glow,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const snow = new THREE.Points(geometry, material);
  snow.frustumCulled = false;
  context.world.add(snow);
  context.update.push((time, delta) => {
    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      positions[offset + 1] -= delta * speeds[index];
      positions[offset] += delta * (0.35 + Math.sin(time * 0.4 + index) * 0.18);
      if (positions[offset + 1] < 0) {
        positions[offset + 1] = 34;
        positions[offset] = rand(-38, 38);
      }
      if (positions[offset] > 38) positions[offset] = -38;
    }
    geometry.attributes.position.needsUpdate = true;
  });
}

function addRain(context: WorldContext, count: number, color = '#c7e2f2'): void {
  const positions = new Float32Array(count * 6);
  for (let index = 0; index < count; index += 1) {
    const x = rand(-17, 17);
    const y = rand(0, 18);
    const z = rand(-18, 8);
    positions.set([x, y, z, x - 0.16, y - 0.65, z], index * 6);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.48,
    depthWrite: false,
  });
  const rain = new THREE.LineSegments(geometry, material);
  context.world.add(rain);
  context.update.push((_time, delta) => {
    for (let index = 0; index < count; index += 1) {
      const offset = index * 6;
      positions[offset + 1] -= delta * 11;
      positions[offset + 4] -= delta * 11;
      if (positions[offset + 1] < 0) {
        positions[offset + 1] = 18;
        positions[offset + 4] = 17.35;
        positions[offset] = rand(-17, 17);
        positions[offset + 3] = positions[offset] - 0.16;
      }
    }
    geometry.attributes.position.needsUpdate = true;
  });
}

function addFireflies(context: WorldContext, count: number, color: string): void {
  const positions = new Float32Array(count * 3);
  const origins = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    origins[index * 3] = rand(-9, 9);
    origins[index * 3 + 1] = rand(0.6, 3.8);
    origins[index * 3 + 2] = rand(-8, 4);
    positions.set(origins.subarray(index * 3, index * 3 + 3), index * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color,
    size: 0.24,
    map: context.glow,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geometry, material);
  context.world.add(points);
  context.update.push((time) => {
    for (let index = 0; index < count; index += 1) {
      positions[index * 3] = origins[index * 3] + Math.sin(time * 0.7 + index) * 0.5;
      positions[index * 3 + 1] = origins[index * 3 + 1] + Math.sin(time * 1.1 + index * 0.7) * 0.35;
    }
    geometry.attributes.position.needsUpdate = true;
  });
}

function addGlow(
  context: WorldContext,
  color: string,
  position: [number, number, number],
  scale: number,
  opacity = 0.65,
): THREE.Sprite {
  const material = new THREE.SpriteMaterial({
    map: context.glow,
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.position.set(...position);
  sprite.scale.set(scale, scale, 1);
  context.world.add(sprite);
  return sprite;
}

function addMoon(context: WorldContext, position: [number, number, number], radius: number): THREE.Group {
  const group = new THREE.Group();
  const moon = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 24, 16),
    new THREE.MeshBasicMaterial({ color: '#fff0c4', fog: false }),
  );
  group.add(moon);
  group.position.set(...position);
  context.world.add(group);
  addGlow(context, '#ffe1a0', position, radius * 7, 0.42);
  return group;
}

/**
 * The reference Antarctica scene works because its aurora is not a ribbon: it
 * is three wide cylindrical curtains wrapped around the viewer. This keeps
 * that geometry and ray shader, with two improvements: a physically cleaner
 * spectral mix (green oxygen core, violet high fringe, faint red veil) and a
 * deterministic twelve-second brightness phrase that travels along each arc.
 */
function addAurora(context: WorldContext): void {
  const LOOP_SECONDS = 12;
  const arcs = [
    {
      seed: 0,
      width: 3.4,
      radius: 34,
      base: 2.6,
      height: 13.6,
      opacity: 1,
      low: new THREE.Vector3(0.35, 1.0, 0.14),
      high: new THREE.Vector3(0.72, 0.22, 1.0),
      crest: 0,
    },
    {
      seed: 0.9,
      width: 2.4,
      radius: 42,
      base: 3.4,
      height: 11.4,
      opacity: 0.82,
      low: new THREE.Vector3(0.30, 0.96, 0.16),
      high: new THREE.Vector3(0.92, 0.24, 0.72),
      crest: 0.3,
    },
    {
      seed: -1,
      width: 2.8,
      radius: 50,
      base: 4.2,
      height: 9.8,
      opacity: 0.62,
      low: new THREE.Vector3(0.44, 1.0, 0.12),
      high: new THREE.Vector3(0.56, 0.34, 1.0),
      crest: 0.6,
    },
  ];

  const vertexShader = `
    uniform float uTime;
    uniform float uLoop;
    uniform float uSeed;
    uniform float uWidth;
    uniform float uRadius;
    uniform float uBase;
    uniform float uHeight;
    varying vec2 vUv;

    void main() {
      vUv = uv;
      float t = 6.2831853 * uTime / uLoop;
      float a = (uv.x - 0.5) * uWidth + uSeed;
      float radius =
        uRadius +
        uSeed * 1.3 +
        sin(uv.x * 7.0 + t + uSeed * 3.0) * 2.2 +
        sin(uv.x * 19.0 - t * 2.0) * 0.8;
      float v = uv.y;
      vec3 p = vec3(
        sin(a) * radius,
        uBase + v * uHeight + sin(uv.x * 11.0 + t + uSeed) * 1.25 * v,
        -cos(a) * radius
      );
      p.x += sin(uv.x * 23.0 + t * 2.0) * 0.64 * v;
      p.z += cos(uv.x * 17.0 - t) * 0.8 * v;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    }
  `;

  const fragmentShader = `
    precision highp float;
    uniform float uTime;
    uniform float uLoop;
    uniform float uSeed;
    uniform float uOpacity;
    uniform float uCrest;
    uniform vec3 uLow;
    uniform vec3 uHigh;
    varying vec2 vUv;

    float cyclicDistance(float a, float b) {
      float d = abs(a - b);
      return min(d, 1.0 - d);
    }

    float crest(float u, float head, float width) {
      float d = cyclicDistance(u, head) / width;
      return exp(-d * d);
    }

    void main() {
      float t = 6.2831853 * uTime / uLoop;
      float phase = fract(uTime / uLoop);
      float u = vUv.x;
      float v = vUv.y;

      // Dense field-aligned rays, with slower folds riding along the curtain.
      float ray =
        pow(0.5 + 0.5 * sin(u * 330.0 + sin(u * 31.0 + t + uSeed) * 5.0), 2.0) *
        (0.6 + 0.4 * sin(u * 53.0 + t + uSeed * 0.5));
      float band =
        0.5 + 0.5 * sin(u * 9.0 - t + uSeed * 5.0) *
        sin(u * 4.0 + t + uSeed);
      float verticalFade =
        smoothstep(0.0, 0.06, v) * pow(1.0 - v, 1.5) * (0.8 + 0.2 * sin(v * 7.0 + t));
      float endFade = smoothstep(0.0, 0.06, u) * smoothstep(1.0, 0.94, u);

      vec3 color = mix(uLow, uHigh, smoothstep(0.55, 0.95, v));
      float redVeil = smoothstep(0.68, 0.96, v) * (1.0 - smoothstep(0.96, 1.0, v)) * 0.18;
      color = mix(color, vec3(0.95, 0.25, 0.35), redVeil);

      // Two soft crests pass through the curtain each twelve-second phrase.
      float travel =
        0.84 +
        crest(u, fract(uCrest + phase), 0.075) * 0.22 +
        crest(u, fract(uCrest + 0.5 - phase), 0.05) * 0.14;

      float alpha = ray * band * verticalFade * endFade * travel * uOpacity * 1.5;
      if (alpha < 0.004) discard;
      gl_FragColor = vec4(color * 1.3, clamp(alpha, 0.0, 0.9));
    }
  `;

  arcs.forEach((arc) => {
    const geometry = new THREE.PlaneGeometry(
      1,
      1,
      context.compact ? 180 : 260,
      context.compact ? 14 : 18,
    );

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uLoop: { value: LOOP_SECONDS },
        uSeed: { value: arc.seed },
        uWidth: { value: arc.width },
        uRadius: { value: arc.radius },
        uBase: { value: arc.base },
        uHeight: { value: arc.height },
        uOpacity: { value: arc.opacity },
        uCrest: { value: arc.crest },
        uLow: { value: arc.low },
        uHigh: { value: arc.high },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: false,
    });
    material.toneMapped = false;

    const curtain = new THREE.Mesh(geometry, material);
    curtain.frustumCulled = false;
    context.world.add(curtain);
    context.update.push((time) => {
      material.uniforms.uTime.value = time;
    });
  });
}

function addWater(
  context: WorldContext,
  options: { color: string; z: number; width?: number; depth?: number; wave?: number; y?: number },
): void {
  const width = options.width ?? 80;
  const depth = options.depth ?? 42;
  const geometry = new THREE.PlaneGeometry(width, depth, 36, 18);
  geometry.rotateX(-Math.PI / 2);
  const material = new THREE.MeshStandardMaterial({
    color: options.color,
    roughness: 0.28,
    metalness: 0.12,
    transparent: true,
    opacity: 0.94,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(0, options.y ?? -0.08, options.z);
  mesh.receiveShadow = !context.compact;
  context.world.add(mesh);
  const base = new Float32Array(geometry.attributes.position.array);
  const wave = options.wave ?? 0.13;
  context.update.push((time) => {
    const position = geometry.attributes.position as THREE.BufferAttribute;
    for (let index = 0; index < position.count; index += 1) {
      const x = base[index * 3];
      const z = base[index * 3 + 2];
      position.setY(
        index,
        Math.sin(x * 0.55 + time * 1.3) * wave + Math.sin(z * 0.85 + time * 0.9) * wave * 0.7,
      );
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();
  });
}

/** Sand with soft dunes, ripples, and warm/cool variation instead of a flat plane. */
function addDuneGround(context: WorldContext, night: boolean): void {
  const geometry = new THREE.PlaneGeometry(80, 24, 48, 18);
  geometry.rotateX(-Math.PI / 2);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const warm = new THREE.Color(night ? '#564268' : '#e9c68d');
  const cool = new THREE.Color(night ? '#342c52' : '#cfa96f');
  const colors: number[] = [];

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const z = position.getZ(index);
    const dune =
      Math.sin(x * 0.22 + z * 0.13) * 0.24 +
      Math.sin(x * 0.61 - z * 0.38) * 0.07 +
      Math.sin(z * 1.4) * 0.025;
    position.setY(index, dune - 0.02);
    const shade = THREE.MathUtils.clamp(0.55 + dune * 0.9 + Math.sin(x * 1.7) * 0.08, 0, 1);
    const color = cool.clone().lerp(warm, shade);
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  position.needsUpdate = true;
  geometry.computeVertexNormals();

  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.98,
      side: THREE.DoubleSide,
    }),
  );
  mesh.position.set(0, 0, 11);
  mesh.receiveShadow = !context.compact;
  context.world.add(mesh);
}

/** Thin, drifting surf lines where the water meets the beach. */
function addShoreFoam(context: WorldContext, night: boolean): void {
  const material = new THREE.LineBasicMaterial({
    color: night ? '#c9c7ee' : '#ffffff',
    transparent: true,
    opacity: night ? 0.42 : 0.68,
    depthWrite: false,
  });
  const lines: THREE.Line[] = [];

  for (let row = 0; row < 4; row += 1) {
    const points: THREE.Vector3[] = [];
    for (let step = 0; step <= 48; step += 1) {
      const x = -16 + (32 * step) / 48;
      const z = -0.8 + row * 0.85 + Math.sin(x * 0.42 + row * 1.7) * 0.28;
      points.push(new THREE.Vector3(x, 0.045 + row * 0.002, z));
    }
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material);
    lines.push(line);
    context.world.add(line);
  }

  context.update.push((time) => {
    lines.forEach((line, index) => {
      line.position.z = Math.sin(time * 0.7 + index * 0.9) * 0.22;
    });
  });
}

/** Small beach props that make the sand read as a place, not a flat plane. */
function addBeachDetails(context: WorldContext, night: boolean): void {
  const starShape = new THREE.Shape();
  for (let index = 0; index <= 10; index += 1) {
    const angle = -Math.PI / 2 + (index / 10) * Math.PI * 2;
    const radius = index % 2 === 0 ? 0.28 : 0.11;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    if (index === 0) starShape.moveTo(x, y);
    else starShape.lineTo(x, y);
  }
  const starGeometry = new THREE.ExtrudeGeometry(starShape, {
    depth: 0.035,
    bevelEnabled: true,
    bevelSize: 0.018,
    bevelThickness: 0.012,
    bevelSegments: 1,
  });
  starGeometry.rotateX(-Math.PI / 2);
  const starfish = new THREE.Mesh(starGeometry, standard(night ? '#d98d78' : '#e9a173', 0.88));
  starfish.position.set(-3.6, 0.09, 6.2);
  starfish.rotation.y = 0.42;
  context.world.add(starfish);

  const shellMaterial = standard(night ? '#d9c6dc' : '#f6e5cf', 0.78);
  const shellRidge = standard(night ? '#a895b6' : '#d4b995', 0.86);
  [
    [1.1, 5.7, 0.3],
    [-0.8, 7.4, -0.5],
    [2.3, 7.2, 0.8],
  ].forEach(([x, z, rotation]) => {
    const shell = new THREE.Group();
    const dome = new THREE.Mesh(new THREE.SphereGeometry(0.17, 14, 8, 0, Math.PI, 0, Math.PI / 2), shellMaterial);
    dome.scale.set(1, 0.48, 0.78);
    shell.add(dome);
    if (!context.compact) {
      for (let ridge = -2; ridge <= 2; ridge += 1) {
        shell.add(
          rodBetween(
            new THREE.Vector3(0, 0.09, 0),
            new THREE.Vector3(ridge * 0.052, 0.035, 0.12 - Math.abs(ridge) * 0.015),
            0.006,
            ridge === 0 ? '#ffffff' : shellRidge.color.getStyle(),
          ),
        );
      }
    }
    shell.position.set(x, 0.06, z);
    shell.rotation.y = rotation;
    context.world.add(shell);
  });

  const towel = new THREE.Mesh(
    new THREE.BoxGeometry(1.55, 0.035, 0.92),
    standard(night ? '#7d6ca2' : '#e9dbc1', 0.95),
  );
  towel.position.set(3.5, 0.09, 5.7);
  towel.rotation.y = -0.22;
  context.world.add(towel);
  for (let stripe = -1; stripe <= 1; stripe += 1) {
    const band = new THREE.Mesh(
      new THREE.BoxGeometry(0.14, 0.012, 0.92),
      standard(night ? '#c6a7d4' : '#cf8062', 0.9),
    );
    band.position.set(stripe * 0.42, 0.112, 0);
    towel.add(band);
  }

  const grassMaterial = standard(night ? '#547060' : '#6d9365', 0.9);
  [-5.8, -5.1, 5.7, 6.4].forEach((x, index) => {
    const z = index < 2 ? 6.5 : 7.8;
    for (let blade = 0; blade < 4; blade += 1) {
      const height = 0.34 + (blade % 3) * 0.11;
      context.world.add(
        rodBetween(
          new THREE.Vector3(x + blade * 0.08, 0.05, z),
          new THREE.Vector3(x + blade * 0.08 + (blade - 1.5) * 0.08, height, z - 0.06),
          0.012,
          grassMaterial.color.getStyle(),
        ),
      );
    }
  });

  [
    [-1.6, 4.6, 1.1],
    [2.1, 7.9, 1.7],
    [6.8, 4.9, 2.4],
  ].forEach(([x, z, seed]) => {
    const stone = new THREE.Mesh(
      organicGeometry(0.2, 1, seed, night ? '#5b5068' : '#9b8874', night ? '#8f819d' : '#d8c8ae', 0.2),
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.98 }),
    );
    stone.position.set(x, 0.11, z);
    stone.scale.set(1.3, 0.65, 1);
    context.world.add(stone);
  });
}

function coloredCanopy(radius: number, colorA: string, colorB: string): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(radius, 32, 8, 0, Math.PI * 2, 0, Math.PI / 2);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const colors: number[] = [];
  const first = new THREE.Color(colorA);
  const second = new THREE.Color(colorB);

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const y = position.getY(index);
    const z = position.getZ(index);
    const theta = Math.atan2(z, x);
    const polar = THREE.MathUtils.clamp(y / radius, 0, 1);
    const scallop = 1 + (1 - polar) * 0.045 * Math.cos(theta * 8);
    position.setXYZ(index, x * scallop, y - (1 - polar) * 0.02, z * scallop);

    const sector = Math.floor(((theta + Math.PI) / (Math.PI * 2)) * 8);
    const color = sector % 2 === 0 ? first : second;
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.72,
    side: THREE.DoubleSide,
  });
  const canopy = new THREE.Mesh(geometry, material);
  canopy.scale.y = 0.42;
  return canopy;
}

function createPenguin(context: WorldContext): THREE.Group {
  const group = new THREE.Group();
  const dark = standard('#20344c', 0.86);
  const white = standard('#f7fbff', 0.9);
  const orange = standard('#e8a33d', 0.75);
  const beakShade = standard('#bf7626', 0.8);
  const ink = standard('#111e2c', 0.38);
  const glint = standard('#ffffff', 0.2);

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.52, 20, 16), dark);
  body.scale.set(0.88, 1.25, 0.78);
  body.position.y = 0.66;
  const belly = new THREE.Mesh(new THREE.SphereGeometry(0.42, 18, 14), white);
  belly.scale.set(0.76, 1.04, 0.42);
  belly.position.set(0, 0.67, 0.35);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.33, 20, 16), dark);
  head.position.y = 1.3;
  const face = new THREE.Mesh(new THREE.SphereGeometry(0.25, 18, 14), white);
  face.scale.set(0.8, 0.9, 0.35);
  face.position.set(0, 1.28, 0.25);

  // Small dark eyes set into the white face, with a single restrained catchlight.
  [-1, 1].forEach((side) => {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.046, 12, 10), ink);
    eye.scale.set(1, 1.12, 0.52);
    eye.position.set(side * 0.145, 1.36, 0.315);
    const spark = new THREE.Mesh(new THREE.SphereGeometry(0.011, 8, 6), glint);
    spark.position.set(side * 0.145 - 0.012, 1.373, 0.344);
    group.add(eye, spark);
  });

  const upperBeak = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.34, 10), orange);
  upperBeak.rotation.x = Math.PI / 2;
  upperBeak.position.set(0, 1.24, 0.49);
  const lowerBeak = new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 8), beakShade);
  lowerBeak.scale.set(0.82, 0.24, 1.15);
  lowerBeak.position.set(0, 1.17, 0.47);

  const leftFlipper = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), dark);
  leftFlipper.scale.set(0.55, 1.9, 0.5);
  leftFlipper.position.set(-0.47, 0.68, 0);
  leftFlipper.rotation.z = 0.28;
  const rightFlipper = leftFlipper.clone();
  rightFlipper.position.x = 0.47;
  rightFlipper.rotation.z = -0.28;
  const leftFoot = new THREE.Mesh(new THREE.SphereGeometry(0.16, 10, 8), orange);
  leftFoot.scale.set(1.1, 0.35, 1.5);
  leftFoot.position.set(-0.22, 0.06, 0.18);
  const rightFoot = leftFoot.clone();
  rightFoot.position.x = 0.22;

  group.add(
    body,
    belly,
    head,
    face,
    upperBeak,
    lowerBeak,
    leftFlipper,
    rightFlipper,
    leftFoot,
    rightFoot,
  );
  group.traverse((child) => {
    if (child instanceof THREE.Mesh && !context.compact) child.castShadow = true;
  });
  group.userData.flippers = [leftFlipper, rightFlipper];
  group.userData.feet = [leftFoot, rightFoot];
  context.world.add(group);
  return group;
}

function createPalm(context: WorldContext): THREE.Group {
  const group = new THREE.Group();
  const bark = standard('#8a653d', 0.95);
  const ringMaterial = standard('#6d4b2d', 0.96);
  const leaf = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.84,
    side: THREE.DoubleSide,
  });
  const veinMaterial = new THREE.LineBasicMaterial({ color: '#2f6747', transparent: true, opacity: 0.72 });
  const coconutMaterial = standard('#7b5535', 0.92);

  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-0.12, 1.1, 0),
    new THREE.Vector3(-0.28, 2.3, 0),
    new THREE.Vector3(-0.1, 3.5, 0),
  ]);
  group.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 22, 0.16, 9, false), bark));

  for (let index = 0; index < 8; index += 1) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.16 - index * 0.006, 0.018, 5, 12), ringMaterial);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(-0.02 - index * 0.03, 0.42 + index * 0.41, 0);
    group.add(ring);
  }

  const crown = new THREE.Group();
  crown.position.set(-0.1, 3.5, 0);
  const frondCount = context.compact ? 6 : 9;
  for (let index = 0; index < frondCount; index += 1) {
    const length = 1.9 + (index % 3) * 0.22;
    const geometry = frondGeometry(length, 0.42);
    const frond = new THREE.Mesh(geometry, leaf);
    frond.rotation.y = (index / frondCount) * Math.PI * 2;
    frond.rotation.z = -0.12 - (index % 3) * 0.11;

    const veinPoints: THREE.Vector3[] = [];
    for (let step = 0; step <= 9; step += 1) {
      const t = step / 9;
      veinPoints.push(
        new THREE.Vector3(t * length, Math.sin(t * Math.PI) * 0.2 - t * t * 0.62 + 0.06, 0),
      );
    }
    const vein = new THREE.Line(new THREE.BufferGeometry().setFromPoints(veinPoints), veinMaterial);
    const branch = new THREE.Group();
    branch.add(frond, vein);
    crown.add(branch);
  }

  [-0.14, 0.12, 0.02].forEach((offset, index) => {
    const coconut = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 10), coconutMaterial);
    coconut.position.set(0.06 + index * 0.04, -0.16 - (index % 2) * 0.05, offset);
    crown.add(coconut);
  });

  group.add(crown);
  group.traverse((child) => {
    if (child instanceof THREE.Mesh && !context.compact) child.castShadow = true;
  });
  context.world.add(group);
  group.userData.crown = crown;
  return group;
}

function createParasol(context: WorldContext, colorA: string, colorB: string): THREE.Group {
  const group = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 2.2, 10), standard('#9a6b42'));
  pole.position.y = 1.1;
  const canopy = coloredCanopy(1.35, colorA, colorB);
  canopy.position.y = 2.1;
  const finial = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 8), standard('#e8c98d'));
  finial.position.y = 2.68;
  group.add(pole, canopy, finial);
  addCanopyDetails(group, 1.35, 2.1, '#b58a5e');
  group.traverse((child) => {
    if (child instanceof THREE.Mesh && !context.compact) child.castShadow = true;
  });
  context.world.add(group);
  return group;
}

function createSailboat(context: WorldContext, night = false): THREE.Group {
  const group = new THREE.Group();
  const hullMaterial = standard(night ? '#c9b48f' : '#f8e6bd', 0.64);
  const trimMaterial = standard(night ? '#8f6cb6' : '#b45f3c', 0.7);
  const deckMaterial = standard(night ? '#d8c2a0' : '#edcf9e', 0.86);
  const wood = standard('#8a653d', 0.86);
  const sailMaterial = new THREE.MeshStandardMaterial({
    color: night ? '#e7b8c7' : '#fdf6e6',
    roughness: 0.78,
    side: THREE.DoubleSide,
  });
  const accent = new THREE.MeshStandardMaterial({
    color: night ? '#8f6cb6' : '#d87845',
    roughness: 0.78,
    side: THREE.DoubleSide,
  });

  const profile = new THREE.Shape();
  profile.moveTo(-1.25, 0.38);
  profile.lineTo(1.25, 0.38);
  profile.quadraticCurveTo(1.5, 0.3, 1.34, 0.06);
  profile.quadraticCurveTo(0.62, -0.3, -0.55, -0.26);
  profile.quadraticCurveTo(-1.16, -0.18, -1.25, 0.38);
  const hullGeometry = new THREE.ExtrudeGeometry(profile, {
    depth: 0.72,
    bevelEnabled: true,
    bevelThickness: 0.06,
    bevelSize: 0.06,
    bevelSegments: 2,
    curveSegments: 10,
  });
  hullGeometry.translate(0, 0, -0.36);
  const hull = new THREE.Mesh(hullGeometry, hullMaterial);
  hull.position.y = 0.16;

  const gunwale = new THREE.Mesh(new THREE.BoxGeometry(2.56, 0.06, 0.82), trimMaterial);
  gunwale.position.y = 0.56;
  const deck = new THREE.Mesh(new THREE.BoxGeometry(2.08, 0.035, 0.6), deckMaterial);
  deck.position.y = 0.49;
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.052, 2.8, 10), wood);
  mast.position.y = 1.85;
  const boom = rodBetween(new THREE.Vector3(0.04, 0.8, 0), new THREE.Vector3(1.42, 0.72, 0), 0.025, wood.color.getStyle());

  const mainSail = new THREE.Mesh(
    sailGeometry(
      new THREE.Vector3(0.08, 3.06, 0),
      new THREE.Vector3(0.08, 0.78, 0),
      new THREE.Vector3(1.42, 0.74, 0),
      0.22,
    ),
    sailMaterial,
  );
  const jib = new THREE.Mesh(
    sailGeometry(
      new THREE.Vector3(-0.05, 2.58, 0),
      new THREE.Vector3(-0.05, 0.76, 0),
      new THREE.Vector3(-1.24, 0.74, 0),
      -0.14,
    ),
    accent,
  );

  const flag = new THREE.Mesh(
    sailGeometry(
      new THREE.Vector3(0.05, 3.3, 0),
      new THREE.Vector3(0.05, 3.05, 0),
      new THREE.Vector3(0.52, 3.17, 0),
      0.06,
    ),
    accent,
  );
  const rigging = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0.05, 3.18, 0),
      new THREE.Vector3(1.34, 0.56, 0),
      new THREE.Vector3(0.05, 3.18, 0),
      new THREE.Vector3(-1.22, 0.56, 0),
      new THREE.Vector3(0.04, 2.4, 0),
      new THREE.Vector3(1.28, 0.74, 0),
    ]),
    new THREE.LineBasicMaterial({ color: '#7b6a52', transparent: true, opacity: 0.65 }),
  );

  group.add(hull, gunwale, deck, mast, boom, mainSail, jib, flag, rigging);
  group.traverse((child) => {
    if (child instanceof THREE.Mesh && !context.compact) child.castShadow = true;
  });
  context.world.add(group);
  return group;
}

function createCrab(context: WorldContext): THREE.Group {
  const group = new THREE.Group();
  const shell = standard('#d87845', 0.72);
  const detail = standard('#b65a35', 0.78);
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 12), shell);
  body.scale.set(1.1, 0.62, 0.85);
  body.position.y = 0.22;
  group.add(body);

  [-1, 1].forEach((side) => {
    const claw = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), shell);
    claw.scale.set(1.25, 0.8, 0.9);
    claw.position.set(side * 0.38, 0.26, 0.1);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), standard('#253043', 0.4));
    eye.position.set(side * 0.13, 0.38, 0.2);
    group.add(claw, eye);
    for (let leg = 0; leg < 3; leg += 1) {
      const limb = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.32, 6), detail);
      limb.rotation.z = side * (0.5 + leg * 0.22);
      limb.position.set(side * (0.2 + leg * 0.05), 0.11, -0.08 - leg * 0.08);
      group.add(limb);
    }
  });
  context.world.add(group);
  return group;
}

function createLantern(context: WorldContext, color: string): THREE.Group {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.2, 0.42, 8),
    new THREE.MeshStandardMaterial({
      color,
      emissive: new THREE.Color(color),
      emissiveIntensity: 1.2,
      roughness: 0.5,
    }),
  );
  const cap = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.16, 8), standard('#6b5541'));
  cap.position.y = 0.29;
  group.add(body, cap);
  group.add(
    new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: context.glow,
        color,
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    ),
  );
  const sprite = group.children[group.children.length - 1] as THREE.Sprite;
  sprite.scale.set(2.2, 2.2, 1);
  context.world.add(group);
  return group;
}

function addGulls(context: WorldContext, count: number): void {
  const positions = new Float32Array(count * 12);
  const origins: Array<[number, number, number]> = [];
  for (let index = 0; index < count; index += 1) {
    origins.push([rand(-8, 8), rand(6, 10), rand(-12, -4)]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.8 });
  const gulls = new THREE.LineSegments(geometry, material);
  context.world.add(gulls);
  context.update.push((time) => {
    origins.forEach(([x, y, z], index) => {
      const offset = time * 0.35 + index;
      const px = x + Math.sin(offset) * 2;
      const py = y + Math.sin(offset * 0.8) * 0.4;
      const flap = Math.sin(time * 2 + index) * 0.16;
      positions.set(
        [px - 0.45, py + flap, z, px, py - 0.08, z, px, py - 0.08, z, px + 0.45, py + flap, z],
        index * 12,
      );
    });
    geometry.attributes.position.needsUpdate = true;
  });
}

/** Layered snow ridges with darker rock at their base, like Antarctic nunataks. */
function ridgeGeometry(
  width: number,
  height: number,
  seed: number,
  night: boolean,
  compact: boolean,
): THREE.BufferGeometry {
  const segments = compact ? 40 : 72;
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const snow = new THREE.Color(night ? '#dcecfa' : '#fbfdff');
  const ice = new THREE.Color(night ? '#6689ab' : '#9fbdd6');
  const rock = new THREE.Color(night ? '#1b3049' : '#53697c');

  for (let index = 0; index <= segments; index += 1) {
    const u = index / segments;
    const x = -width / 2 + width * u;
    const broad = Math.abs(Math.sin(u * Math.PI * 2.2 + seed));
    const peaks = Math.abs(Math.sin(u * Math.PI * 6.4 + seed * 1.7));
    const summits = Math.pow(Math.abs(Math.sin(u * Math.PI * 3.7 + seed * 0.8)), 8) * 0.28;
    const detail = Math.sin(u * Math.PI * 17 + seed * 2.3) * 0.035;
    const taper = 0.7 + Math.sin(Math.PI * u) * 0.3;
    const topY = height * (0.2 + broad * 0.44 + peaks * 0.2 + summits + detail) * taper;
    const baseY = -0.45;
    const midY = baseY + (topY - baseY) * 0.46;
    positions.push(x, topY, 0, x, midY, 0.1, x, baseY, 0.18);
    colors.push(snow.r, snow.g, snow.b, ice.r, ice.g, ice.b, rock.r, rock.g, rock.b);
  }

  for (let index = 0; index < segments; index += 1) {
    const a = index * 3;
    const b = (index + 1) * 3;
    indices.push(a, a + 1, b, a + 1, b + 1, b);
    indices.push(a + 1, a + 2, b + 1, a + 2, b + 2, b + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function addMountainRange(context: WorldContext, night: boolean): void {
  [
    { width: 180, height: 16, x: 0, z: -52, seed: 1.12, opacity: 0.92 },
    { width: 160, height: 11.5, x: 0, z: -41, seed: 3.74, opacity: 0.96 },
    { width: 140, height: 6.8, x: 0, z: -31, seed: 5.38, opacity: 1 },
    // A clearer massif behind the colony reads as a real background mountain.
    { width: 54, height: 13.5, x: -9, z: -38, seed: 2.24, opacity: 1 },
    { width: 40, height: 9.2, x: 22, z: -43, seed: 4.91, opacity: 0.98 },
  ].forEach((layer) => {
    const ridge = new THREE.Mesh(
      ridgeGeometry(layer.width, layer.height, layer.seed, night, context.compact),
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.98,
        side: THREE.DoubleSide,
        transparent: layer.opacity < 1,
        opacity: layer.opacity,
      }),
    );
    ridge.position.set(layer.x, 0, layer.z);
    context.world.add(ridge);
  });

  // Nunataks: exposed rock pushing through the ice near the mountain front.
  [
    [-18, -33.5, 1.05, 2.2],
    [-7.5, -36, 0.78, 1.4],
    [11.5, -34.2, 1, 3.1],
    [21, -37, 0.82, 4.4],
  ].forEach(([x, z, scale, seed]) => {
    const rock = new THREE.Mesh(
      organicGeometry(
        0.8,
        1,
        seed,
        night ? '#243a53' : '#5b6d7d',
        night ? '#4c6985' : '#93a4b2',
        0.26,
      ),
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }),
    );
    rock.position.set(x, 0.32, z);
    rock.scale.set(scale * 1.35, scale * 1.55, scale);
    context.world.add(rock);
  });
}

/** Deterministic value noise used for the polar night terrain, like the reference. */
function polarHash(x: number, z: number): number {
  const value = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return value - Math.floor(value);
}

function polarNoise(x: number, z: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const sx = fx * fx * (3 - 2 * fx);
  const sz = fz * fz * (3 - 2 * fz);
  const a = polarHash(ix, iz);
  const b = polarHash(ix + 1, iz);
  const c = polarHash(ix, iz + 1);
  const d = polarHash(ix + 1, iz + 1);
  return (a * (1 - sx) + b * sx) * (1 - sz) + (c * (1 - sx) + d * sx) * sz;
}

function polarFbm(x: number, z: number): number {
  let total = 0;
  let amplitude = 0.5;
  for (let octave = 0; octave < 5; octave += 1) {
    total += amplitude * polarNoise(x, z);
    x = x * 2.03 + 11;
    z = z * 2.03 + 7;
    amplitude *= 0.5;
  }
  return total;
}

function polarGroundHeight(x: number, z: number, night: boolean): number {
  const flow = Math.sin(x * 0.16 + z * 0.38) * 0.045 + Math.sin(x * 0.63 - z * 0.21) * 0.014;
  const scour = Math.sin(z * 0.92 + x * 0.07) * 0.012;
  if (!night) {
    // Daylit fast ice is not a flat sheet: low swells, wind-carved sastrugi,
    // and lee drifts give the colony a real surface to stand on.
    const swell = Math.sin(x * 0.085 + z * 0.12) * 0.13 + Math.sin(x * 0.2 - z * 0.075) * 0.055;
    const sastrugi = Math.sin(x * 0.86 + z * 0.34) * 0.02 + Math.sin(x * 1.7 - z * 0.58) * 0.009;
    const drift = Math.max(0, Math.sin((x + z * 0.42) * 0.17)) * 0.075;
    const grains = (polarFbm(x * 0.42 + 13, z * 0.42 + 5) - 0.5) * 0.035;
    return flow + scour + swell + sastrugi + drift + grains - 0.12;
  }
  const broad = polarFbm(x * 0.035 + 7, z * 0.035 + 3);
  const ridge = 1 - Math.abs(2 * polarFbm(x * 0.02 + 30, z * 0.02 + 9) - 1);
  const basinDistance = Math.hypot((x - 2) / 23, (z - 16) / 7);
  const basin = 1 - THREE.MathUtils.smoothstep(basinDistance, 0.72, 1.12);
  return flow + scour + broad * 0.28 + Math.pow(ridge, 3) * 0.5 - 0.1 - basin * 0.55;
}

/** Wind-scoured fast ice with flow lines, crevasses, and blue-ice patches. */
function addIceShelf(context: WorldContext, night: boolean): void {
  const geometry = new THREE.PlaneGeometry(
    80,
    58,
    context.compact ? 40 : 64,
    context.compact ? 24 : 40,
  );
  geometry.rotateX(-Math.PI / 2);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const fresh = new THREE.Color(night ? '#d7e9f8' : '#ffffff');
  const blue = new THREE.Color(night ? '#608fb6' : '#b8d7e9');
  const colors: number[] = [];

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const z = position.getZ(index);
    const broad = polarFbm(x * 0.035 + 7, z * 0.035 + 3);
    const y = polarGroundHeight(x, z, night);
    position.setY(index, y);
    const shade = THREE.MathUtils.clamp(
      night
        ? 0.24 + y * 1.15 + broad * 0.34 + Math.sin(x * 0.34 - z * 0.18) * 0.08
        : 0.55 + y * 5 + Math.sin(x * 0.34 - z * 0.18) * 0.12,
      0,
      1,
    );
    const color = blue.clone().lerp(fresh, shade);
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  position.needsUpdate = true;
  geometry.computeVertexNormals();

  const ground = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.02 }),
  );
  ground.position.z = -2;
  ground.receiveShadow = !context.compact;
  context.world.add(ground);

  const flowMaterial = new THREE.MeshBasicMaterial({
    color: night ? '#4d7da6' : '#9dc3dc',
    transparent: true,
    opacity: night ? 0.26 : 0.22,
    depthWrite: false,
  });
  for (let index = 0; index < 9; index += 1) {
    const x = -18 + index * 4.4;
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x + 1.4, 0.02, 15),
      new THREE.Vector3(x + 0.4, 0.02, 5),
      new THREE.Vector3(x - 0.6, 0.02, -6),
      new THREE.Vector3(x - 1.7, 0.02, -16),
      new THREE.Vector3(x - 2.4, 0.02, -26),
    ]);
    context.world.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 30, 0.012, 4, false), flowMaterial));
  }

  const crevasseMaterial = new THREE.MeshBasicMaterial({
    color: night ? '#1e4266' : '#668eaf',
    transparent: true,
    opacity: night ? 0.58 : 0.44,
    depthWrite: false,
  });
  [
    [-13, -17, 2.9],
    [-4.5, -21, 3.4],
    [4.5, -15, 2.7],
    [13, -20, 3.1],
    [-9, -6, 2.2],
    [8, -7, 2.5],
  ].forEach(([x, z, length], index) => {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x - length / 2, 0.035, z - 0.3),
      new THREE.Vector3(x - length * 0.15, 0.035, z + 0.18),
      new THREE.Vector3(x + length * 0.2, 0.035, z - 0.12),
      new THREE.Vector3(x + length / 2, 0.035, z + 0.34),
    ]);
    context.world.add(
      new THREE.Mesh(new THREE.TubeGeometry(curve, 18, 0.018, 4, false), crevasseMaterial),
    );
    if (index % 2 === 0) {
      const branch = new THREE.CatmullRomCurve3([
        new THREE.Vector3(x - length * 0.05, 0.035, z + 0.15),
        new THREE.Vector3(x + length * 0.08, 0.035, z + 0.85),
        new THREE.Vector3(x + length * 0.24, 0.035, z + 1.25),
      ]);
      context.world.add(
        new THREE.Mesh(new THREE.TubeGeometry(branch, 12, 0.012, 4, false), crevasseMaterial),
      );
    }
  });

  const patchMaterial = new THREE.MeshBasicMaterial({
    color: night ? '#7ba8ce' : '#a9cee5',
    transparent: true,
    opacity: night ? 0.24 : 0.3,
    depthWrite: false,
  });
  [
    [-12.5, 3.5, 3.2, 1.1, 0.2],
    [-2, -9, 2.6, 0.9, -0.35],
    [9.5, 1.5, 3.6, 1.3, 0.15],
    [15.5, -12, 2.4, 0.8, -0.2],
    [-17, -5, 2.2, 0.7, 0.4],
  ].forEach(([x, z, width, depth, angle]) => {
    const patchGeometry = new THREE.CircleGeometry(1, 36);
    patchGeometry.rotateX(-Math.PI / 2);
    const patch = new THREE.Mesh(patchGeometry, patchMaterial);
    patch.position.set(x, 0.018, z);
    patch.scale.set(width, 1, depth);
    patch.rotation.y = angle;
    context.world.add(patch);
  });
}

function shelfFrontGeometry(width: number, night: boolean, compact: boolean): THREE.BufferGeometry {
  const segments = compact ? 30 : 52;
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const top = new THREE.Color(night ? '#d3e5f5' : '#ffffff');
  const face = new THREE.Color(night ? '#6d99bf' : '#b7d5e8');
  const deep = new THREE.Color(night ? '#2f587b' : '#7ea8c6');

  for (let index = 0; index <= segments; index += 1) {
    const u = index / segments;
    const x = -width / 2 + width * u;
    const crest =
      0.48 + Math.sin(u * Math.PI * 6.1 + 1.7) * 0.09 + Math.sin(u * Math.PI * 13 + 0.6) * 0.025;
    const mid = 0.06 + Math.sin(u * Math.PI * 8.3 + 2.1) * 0.035;
    positions.push(x, crest, 0, x, mid, 0.12, x, -0.32, 0.2);
    colors.push(top.r, top.g, top.b, face.r, face.g, face.b, deep.r, deep.g, deep.b);
  }

  for (let index = 0; index < segments; index += 1) {
    const a = index * 3;
    const b = (index + 1) * 3;
    indices.push(a, a + 1, b, a + 1, b + 1, b);
    indices.push(a + 1, a + 2, b + 1, a + 2, b + 2, b + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function addShelfFront(context: WorldContext, night: boolean): void {
  const front = new THREE.Mesh(
    shelfFrontGeometry(78, night, context.compact),
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.86, side: THREE.DoubleSide }),
  );
  front.position.z = -25;
  front.receiveShadow = !context.compact;
  context.world.add(front);

  const crackMaterial = new THREE.MeshBasicMaterial({
    color: night ? '#1b3f61' : '#5d87a8',
    transparent: true,
    opacity: night ? 0.5 : 0.38,
    depthWrite: false,
  });
  [-24, -14, -5, 6, 16, 25].forEach((x, index) => {
    const drift = index % 2 ? 0.12 : -0.12;
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x, 0.42, -24.78),
      new THREE.Vector3(x + drift, 0.23, -24.76),
      new THREE.Vector3(x + drift * 1.5, 0.03, -24.73),
    ]);
    context.world.add(
      new THREE.Mesh(new THREE.TubeGeometry(curve, 10, 0.012, 4, false), crackMaterial),
    );
  });
}

function addAntarcticBackdrop(context: WorldContext, night: boolean): void {
  addMountainRange(context, night);
  addIceShelf(context, night);
  addShelfFront(context, night);
}

/** One shared direction, separate wide lanes, and a seamless off-screen wrap. */
function addPenguinWalkers(context: WorldContext, night: boolean, scaleFactor = 1): void {
  const baseLanes = [
    { z: -12.5, scale: 0.5, offset: 0 },
    { z: -7.8, scale: 0.62, offset: 10 },
    { z: -3.4, scale: 0.74, offset: 20 },
    { z: -15, scale: 0.46, offset: 30 },
    { z: 2.6, scale: 1.02, offset: 40 },
    { z: -10.2, scale: 0.58, offset: 50 },
    { z: -5.2, scale: 0.8, offset: 60 },
    { z: -18.4, scale: 0.42, offset: 70 },
    { z: 6.2, scale: 0.88, offset: 80 },
    { z: -0.8, scale: 0.67, offset: 90 },
    { z: -21.2, scale: 0.38, offset: 100 },
    { z: 4.4, scale: 0.76, offset: 110 },
  ];
  // A few more walkers fill the line without bunching it together.
  const extraLanes = [
    { z: -20.6, scale: 0.45, offset: 101 },
    { z: -16.4, scale: 0.54, offset: 107 },
    { z: -11.6, scale: 0.68, offset: 112 },
    { z: -7.2, scale: 0.8, offset: 117 },
    { z: -13.2, scale: 0.6, offset: 122 },
  ];
  const lanes = [
    ...baseLanes.slice(0, context.compact ? 7 : 12),
    ...extraLanes,
  ];
  const startX = -38;
  const span = 76;
  const speed = 0.55;

  const walkers = lanes.map((lane, index) => {
    const penguin = createPenguin(context);
    penguin.scale.setScalar(lane.scale * scaleFactor);
    penguin.rotation.y = Math.PI / 2;
    penguin.position.set(startX + lane.offset, 0, lane.z);
    return { penguin, lane, index };
  });

  context.update.push((time) => {
    walkers.forEach(({ penguin, lane, index }) => {
      const travelled = (time * speed + lane.offset) % span;
      const step = time * 3.1 + index * 0.55;
      penguin.position.x = startX + travelled;
      penguin.position.y = polarGroundHeight(penguin.position.x, penguin.position.z, night) +
        Math.abs(Math.sin(step)) * 0.035;
      penguin.rotation.y = Math.PI / 2 + Math.sin(time * 0.45 + index) * 0.025;
      penguin.rotation.z = Math.sin(step) * 0.035;

      const flippers = penguin.userData.flippers as THREE.Mesh[] | undefined;
      flippers?.forEach((flipper, flipperIndex) => {
        const base = flipperIndex === 0 ? 0.28 : -0.28;
        flipper.rotation.z = base + Math.sin(step) * (flipperIndex === 0 ? 0.1 : -0.1);
      });
      const feet = penguin.userData.feet as THREE.Mesh[] | undefined;
      feet?.forEach((foot, footIndex) => {
        foot.position.z = 0.18 + Math.sin(step + footIndex * Math.PI) * 0.07;
      });
    });
  });
}

/**
 * A petrel wing section: broad at the shoulder, narrowing to a rounded tip,
 * swept slightly back, and softly cambered so it does not read as a rectangle.
 */
function petrelWingGeometry(length: number, rootChord: number, sweep: number): THREE.BufferGeometry {
  const segments = 12;
  const positions: number[] = [];
  const indices: number[] = [];
  for (let index = 0; index <= segments; index += 1) {
    const t = index / segments;
    const taper = Math.pow(1 - t, 0.62);
    const chord = Math.max(0.035, rootChord * taper);
    const x = length * t;
    const camber = Math.sin(t * Math.PI) * 0.04;
    const trailing = -sweep * t * t;
    positions.push(
      x, camber, trailing - chord * 0.4,
      x, camber, trailing + chord * 0.6,
    );
    if (index < segments) {
      const a = index * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Inner wing and outer hand/wrist as separate pivots. The wrist lags the
 * shoulder slightly, which is what makes a gull or petrel read as folding
 * through the air instead of flapping like a flat board.
 */
function createPetrelWing(material: THREE.Material): {
  group: THREE.Group;
  shoulder: THREE.Group;
  wrist: THREE.Group;
} {
  const group = new THREE.Group();
  const shoulder = new THREE.Group();

  const inner = new THREE.Mesh(petrelWingGeometry(1.05, 0.72, 0.06), material);
  shoulder.add(inner);

  const wrist = new THREE.Group();
  wrist.position.x = 1.02;
  const outer = new THREE.Mesh(petrelWingGeometry(0.78, 0.4, 0.24), material);
  outer.position.z = 0.06;
  wrist.add(outer);
  shoulder.add(wrist);

  group.add(shoulder);
  return { group, shoulder, wrist };
}

/** Snow petrels circling above the colony, with individually flapping wings. */
function addAntarcticPetrels(context: WorldContext, night: boolean): void {
  const ink = standard('#05080d', 0.3);
  const white = standard('#f4f4f4', 0.5);
  const wingMaterial = new THREE.MeshStandardMaterial({
    color: '#f4f7fb',
    roughness: 0.9,
    side: THREE.DoubleSide,
  });
  const placement = night
    ? { radius: 14, step: 3, centerX: -6, centerZ: 6, height: 11, rise: 1.5, bob: 1.2 }
    : { radius: 7.5, step: 1.4, centerX: 0, centerZ: -12, height: 4.8, rise: 0.45, bob: 0.55 };
  const birds = Array.from({ length: context.compact ? 3 : 5 }, (_, index) => {
    const bird = new THREE.Group();
    const body = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 10), white);
    body.scale.set(0.22, 0.2, 0.6);
    const head = new THREE.Mesh(new THREE.SphereGeometry(1, 10, 8), white);
    head.scale.setScalar(0.15);
    head.position.set(0, 0.05, 0.6);
    const beak = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.28, 8), ink);
    beak.rotation.x = Math.PI / 2;
    beak.position.set(0, 0.03, 0.82);
    const right = createPetrelWing(wingMaterial);
    const left = createPetrelWing(wingMaterial);
    // Mirror across the body only on X. Rotating by PI would also flip Z,
    // making the left wing's leading edge face backwards.
    left.group.scale.x = -1;
    bird.add(body, head, beak, right.group, left.group);
    context.world.add(bird);
    return { bird, left, right, index };
  });

  context.update.push((time) => {
    birds.forEach(({ bird, left, right, index }) => {
      const angle = time * (0.22 + index * 0.02) + index * 1.3;
      const radius = placement.radius + index * placement.step;
      bird.position.set(
        placement.centerX + Math.cos(angle) * radius,
        placement.height + index * placement.rise + Math.sin(time * 0.7 + index) * placement.bob,
        placement.centerZ + Math.sin(angle) * radius,
      );
      bird.rotation.y = Math.atan2(-Math.sin(angle), Math.cos(angle));

      // Mirroring on X keeps both wings' leading edges pointing forward, so
      // the same local rotation.z sign raises both shoulders together. A small
      // delayed wrist bend softens the silhouette at each end of the stroke.
      const stroke = time * 5.6 + index * 0.85;
      const shoulderAngle = Math.sin(stroke) * 0.34 - 0.06;
      const wristAngle = Math.sin(stroke - 0.7) * 0.26 - 0.12;
      [left, right].forEach((wing) => {
        wing.shoulder.rotation.z = shoulderAngle;
        wing.wrist.rotation.z = wristAngle;
      });
    });
  });
}

/** Seal, orca, and rounded icebergs shared by the polar day and night worlds. */
function addAntarcticWildlife(context: WorldContext, night: boolean): void {
  const fur = standard(night ? '#6b6661' : '#7d776f', 0.92);
  const paleFur = standard(night ? '#a39d94' : '#b9b2a8', 0.92);
  const ink = standard('#05080d', 0.3);
  const black = standard(night ? '#0b0d12' : '#101821', 0.42);
  const white = standard(night ? '#f4f4f4' : '#f7f7f2', 0.5);
  // Daylight framing is a slow orbit, so the large animals are kept nearer
  // their real scale relative to the penguin colony.
  const sealScale = night ? 1 : 0.78;
  const orcaScale = night ? 1 : 0.7;

  // Weddell seal resting on the near ice, breathing slowly.
  const seal = new THREE.Group();
  const sealBody = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 14), fur);
  sealBody.scale.set(2.3, 0.8, 0.85);
  sealBody.position.y = 0.8;
  const sealBelly = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), paleFur);
  sealBelly.scale.set(2.0, 0.5, 0.7);
  sealBelly.position.set(0, 0.57, 0.1);
  const sealHead = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), fur);
  sealHead.scale.set(0.7, 0.55, 0.55);
  sealHead.position.set(2.2, 1.0, 0);
  const sealSnout = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 10), paleFur);
  sealSnout.scale.set(0.3, 0.25, 0.3);
  sealSnout.position.set(2.8, 0.9, 0);
  seal.add(sealBody, sealBelly, sealHead, sealSnout);
  [-0.3, 0.3].forEach((z) => {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 10, 8), ink);
    eye.position.set(2.55, 1.25, z);
    seal.add(eye);
  });
  [-0.45, 0.45].forEach((z, index) => {
    const flipper = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), fur);
    flipper.scale.set(0.75, 0.12, 0.28);
    flipper.position.set(-1.8, 0.45, z);
    flipper.rotation.y = index === 0 ? 0.4 : -0.4;
    seal.add(flipper);
  });
  const sealX = night ? -8 : -3;
  const sealZ = night ? 8 : -7.5;
  seal.position.set(sealX, polarGroundHeight(sealX, sealZ, night) + 0.02, sealZ);
  seal.rotation.y = 0.6;
  seal.scale.setScalar(sealScale);
  context.world.add(seal);

  // A distant orca breaks the meltwater edge.
  const orca = new THREE.Group();
  const orcaBody = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 14), black);
  orcaBody.scale.set(3.4, 0.95, 0.95);
  const orcaBelly = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), white);
  orcaBelly.scale.set(1.6, 0.6, 0.8);
  orcaBelly.position.set(0.4, -0.35, 0);
  const dorsal = new THREE.Mesh(new THREE.ConeGeometry(0.5, 2, 12), black);
  dorsal.position.set(0.3, 1.3, 0);
  dorsal.rotation.z = 0.3;
  const tail = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), black);
  tail.scale.set(0.5, 0.1, 1.2);
  tail.position.x = -3.4;
  orca.add(orcaBody, orcaBelly, dorsal, tail);
  orca.scale.setScalar(orcaScale);
  context.world.add(orca);

  // Rounded, faceted icebergs instead of flat tabular slabs.
  const bergs = [
    { x: -12, z: -20, s: 1.6, seed: 1.2 },
    { x: 14, z: -22, s: 2.2, seed: 2.8 },
    { x: -18, z: -28, s: 1.9, seed: 4.4 },
    { x: 8, z: -28, s: 1.4, seed: 6.1 },
  ].map((config) => {
    const berg = new THREE.Mesh(
      organicGeometry(
        1,
        1,
        config.seed,
        night ? '#5f8db2' : '#7fa9c8',
        night ? '#d9f2ff' : '#f2fbff',
        0.22,
      ),
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.7,
        metalness: 0.04,
        flatShading: true,
      }),
    );
    berg.scale.set(config.s, config.s * 0.82, config.s);
    berg.position.set(config.x, config.s * 0.32, config.z);
    context.world.add(berg);
    return { berg, baseY: config.s * 0.32, seed: config.seed };
  });

  const waterZ = night ? 11 : -22;
  const centerX = night ? 2 : 0;
  const swimX = night ? 8 : 6;
  const swimZ = night ? 3.2 : 2.4;
  context.update.push((time) => {
    seal.scale.y = sealScale * (1 + Math.sin(time * 1.6) * 0.03);
    const swim = time * 0.18;
    orca.position.set(
      centerX + Math.cos(swim) * swimX,
      -0.45 + Math.sin(time * 0.8) * 0.18,
      waterZ + Math.sin(swim) * swimZ,
    );
    orca.rotation.y = Math.atan2(-Math.cos(swim) * swimZ, -Math.sin(swim) * swimX);
    orca.rotation.z = Math.cos(time * 0.8) * 0.1;
    bergs.forEach(({ berg, baseY, seed }) => {
      berg.position.y = baseY + Math.sin(time * 0.6 + seed) * 0.12;
      berg.rotation.y = Math.sin(time * 0.12 + seed) * 0.18;
    });
  });
}

const DAY_LIGHT_POSITION: [number, number, number] = [-34, 9, 8];

/** Overcast-feeling polar daylight: cool gradient and thin ice-cloud bands, with no sun in view. */
function addAntarcticDaySky(context: WorldContext): void {
  const material = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vDirection;
      void main() {
        vDirection = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec3 vDirection;
      void main() {
        vec3 d = normalize(vDirection);
        float h = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 zenith = vec3(0.36, 0.59, 0.78);
        vec3 mid = vec3(0.79, 0.88, 0.94);
        vec3 horizon = vec3(0.97, 0.96, 0.93);
        vec3 color = mix(horizon, mid, smoothstep(0.48, 0.72, h));
        color = mix(color, zenith, smoothstep(0.68, 1.0, h));
        color += vec3(0.05, 0.06, 0.07) * exp(-abs(d.y) * 9.0);
        float cirrus = sin(d.x * 17.0 + d.z * 9.0 + sin(d.y * 23.0) * 1.6) * 0.5 + 0.5;
        cirrus *= smoothstep(0.1, 0.65, d.y) * (1.0 - smoothstep(0.75, 1.0, d.y));
        color += vec3(0.05, 0.06, 0.07) * pow(cirrus, 6.0) * 0.3;
        gl_FragColor = vec4(color, 1.0);
      }
    `,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
  });
  material.toneMapped = false;
  const sky = new THREE.Mesh(new THREE.SphereGeometry(150, 40, 24), material);
  sky.renderOrder = -1;
  context.world.add(sky);
}

/** A five-band mountain profile: snow cap, ice, rock, scree, and shadow. */
function dayRidgeGeometry(
  width: number,
  height: number,
  seed: number,
  compact: boolean,
): THREE.BufferGeometry {
  const segments = compact ? 44 : 80;
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const snow = new THREE.Color('#fbfdff');
  const ice = new THREE.Color('#a7c4da');
  const rock = new THREE.Color('#5b6d7d');
  const deep = new THREE.Color('#31465b');

  for (let index = 0; index <= segments; index += 1) {
    const u = index / segments;
    const x = -width / 2 + width * u;
    const broad = Math.abs(Math.sin(u * Math.PI * 2.1 + seed));
    const peaks = Math.pow(Math.abs(Math.sin(u * Math.PI * 5.3 + seed * 1.4)), 6) * 0.34;
    const serration = Math.abs(Math.sin(u * Math.PI * 13 + seed * 2.1)) * 0.06;
    const detail = Math.sin(u * Math.PI * 29 + seed * 3.7) * 0.018;
    const taper = 0.58 + Math.sin(Math.PI * u) * 0.42;
    const top = height * (0.16 + broad * 0.46 + peaks + serration + detail) * taper;
    const bands: Array<{ y: number; z: number; color: THREE.Color }> = [
      { y: top, z: 0, color: snow },
      { y: top * 0.76, z: 0.08, color: snow.clone().lerp(ice, 0.25) },
      { y: top * 0.46, z: 0.16, color: ice },
      { y: top * 0.17, z: 0.24, color: rock },
      { y: -0.55, z: 0.32, color: deep },
    ];
    bands.forEach(({ y, z, color }) => {
      positions.push(x, y, z);
      colors.push(color.r, color.g, color.b);
    });
  }

  for (let index = 0; index < segments; index += 1) {
    for (let band = 0; band < 4; band += 1) {
      const a = index * 5 + band;
      const b = (index + 1) * 5 + band;
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Mountains ring the basin so a full orbit never reveals an empty horizon. */
function addAntarcticDayMountains(context: WorldContext): void {
  [
    { angle: 0, radius: 76, width: 96, height: 20, seed: 1.2 },
    { angle: 0.7, radius: 74, width: 72, height: 15, seed: 2.6 },
    { angle: 1.4, radius: 78, width: 84, height: 18, seed: 3.9 },
    { angle: 2.1, radius: 72, width: 68, height: 13, seed: 5.1 },
    { angle: 2.8, radius: 76, width: 88, height: 17, seed: 6.4 },
    { angle: -0.7, radius: 74, width: 76, height: 16, seed: 7.8 },
    { angle: -1.4, radius: 78, width: 90, height: 19, seed: 9.2 },
    { angle: -2.1, radius: 72, width: 66, height: 12, seed: 10.5 },
    { angle: -2.8, radius: 76, width: 86, height: 17, seed: 11.7 },
  ].forEach(({ angle, radius, width, height, seed }) => {
    const ridge = new THREE.Mesh(
      dayRidgeGeometry(width, height, seed, context.compact),
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 1,
        side: THREE.DoubleSide,
      }),
    );
    ridge.position.set(Math.sin(angle) * radius, -0.4, -Math.cos(angle) * radius);
    ridge.rotation.y = -angle;
    context.world.add(ridge);
  });

  // Exposed rock close to the colony: nunataks with wind-polished snowcaps.
  [
    [-16, -26, 1.25, 2.2],
    [9, -28, 1.05, 4.7],
    [20, -19, 0.82, 7.3],
    [-22, -17, 0.92, 9.6],
  ].forEach(([x, z, scale, seed]) => {
    const rock = new THREE.Mesh(
      organicGeometry(0.9, 1, seed, '#46596b', '#9fb1bf', 0.28),
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, flatShading: true }),
    );
    const y = polarGroundHeight(x, z, false);
    rock.position.set(x, y + scale * 0.28, z);
    rock.scale.set(scale * 1.35, scale * 1.55, scale);
    context.world.add(rock);

    const cap = new THREE.Mesh(
      organicGeometry(0.62, 1, seed + 1.5, '#dceaf5', '#ffffff', 0.18),
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, flatShading: true }),
    );
    cap.position.set(x, y + scale * 0.95, z);
    cap.scale.set(scale * 1.1, scale * 0.42, scale * 0.95);
    context.world.add(cap);
  });
}

/** A wide, wind-carved ice surface with sastrugi, crevasses, and blue ice. */
function addAntarcticDayTerrain(context: WorldContext): void {
  const segments = context.compact ? 48 : 96;
  const geometry = new THREE.PlaneGeometry(160, 160, segments, segments);
  geometry.rotateX(-Math.PI / 2);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const fresh = new THREE.Color('#ffffff');
  const blue = new THREE.Color('#a8c8de');
  const colors: number[] = [];

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const z = position.getZ(index);
    const y = polarGroundHeight(x, z, false);
    position.setY(index, y);
    const grain = polarFbm(x * 0.42 + 13, z * 0.42 + 5);
    const exposure = THREE.MathUtils.clamp(
      0.55 + y * 2.1 + (grain - 0.5) * 0.7 + Math.sin(x * 0.3 - z * 0.14) * 0.12,
      0,
      1,
    );
    const color = blue.clone().lerp(fresh, exposure);
    if (Math.sin(x * 0.7 + z * 0.25) > 0.86) color.lerp(blue, 0.22);
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  const ground = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0.02 }),
  );
  ground.receiveShadow = !context.compact;
  context.world.add(ground);

  const flowMaterial = new THREE.MeshBasicMaterial({
    color: '#9cc2db',
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
  });
  for (let index = 0; index < 14; index += 1) {
    const baseX = -42 + index * 6.4;
    const points = [
      [baseX + 1.8, 26],
      [baseX + 0.8, 12],
      [baseX - 0.4, -2],
      [baseX - 1.5, -16],
      [baseX - 2.4, -34],
    ].map(([x, z]) => new THREE.Vector3(x, polarGroundHeight(x, z, false) + 0.025, z));
    context.world.add(
      new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 28, 0.014, 4, false), flowMaterial),
    );
  }

  const crevasseMaterial = new THREE.MeshBasicMaterial({
    color: '#5a85a6',
    transparent: true,
    opacity: 0.52,
    depthWrite: false,
  });
  [
    [-15, -18, 3.4],
    [-5, -23, 3.8],
    [5, -17, 3.1],
    [15, -22, 3.6],
    [-10, -7, 2.6],
    [9, -8, 2.9],
    [-24, -31, 3.2],
    [24, -30, 2.8],
  ].forEach(([x, z, length], index) => {
    const points = [
      [x - length / 2, z - 0.35],
      [x - length * 0.15, z + 0.2],
      [x + length * 0.2, z - 0.15],
      [x + length / 2, z + 0.38],
    ].map(([px, pz]) => new THREE.Vector3(px, polarGroundHeight(px, pz, false) + 0.03, pz));
    context.world.add(
      new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 20, 0.02, 4, false), crevasseMaterial),
    );
    if (index % 2 === 0) {
      const branch = [
        [x, z + 0.18],
        [x + length * 0.12, z + 0.85],
        [x + length * 0.3, z + 1.35],
      ].map(([px, pz]) => new THREE.Vector3(px, polarGroundHeight(px, pz, false) + 0.03, pz));
      context.world.add(
        new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(branch), 12, 0.012, 4, false), crevasseMaterial),
      );
    }
  });

  const patchMaterial = new THREE.MeshBasicMaterial({
    color: '#9fc6de',
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
  });
  [
    [-13, 4, 3.4, 1.2, 0.2],
    [-2, -10, 2.8, 0.9, -0.35],
    [10, 2, 3.8, 1.35, 0.15],
    [16, -13, 2.6, 0.85, -0.2],
    [-18, -5, 2.4, 0.75, 0.4],
  ].forEach(([x, z, width, depth, angle]) => {
    const patch = new THREE.Mesh(new THREE.CircleGeometry(1, 36).rotateX(-Math.PI / 2), patchMaterial);
    patch.position.set(x, polarGroundHeight(x, z, false) + 0.025, z);
    patch.scale.set(width, 1, depth);
    patch.rotation.y = angle;
    context.world.add(patch);
  });

  // Wind slabs: low, elongated drifts that catch the daylight.
  [
    [-24, 2, 2.8, 0.5, 0.25],
    [-8, 13, 2.2, 0.4, -0.15],
    [7, 9, 3.1, 0.5, 0.1],
    [22, 0, 2.4, 0.42, -0.3],
    [14, -27, 2.6, 0.46, 0.18],
  ].forEach(([x, z, length, height, angle], index) => {
    const drift = new THREE.Mesh(
      organicGeometry(1, 1, 20 + index * 3, '#cfe2f0', '#ffffff', 0.16),
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.96 }),
    );
    drift.position.set(x, polarGroundHeight(x, z, false) + height * 0.34, z);
    drift.scale.set(length, height, 0.75);
    drift.rotation.y = angle;
    context.world.add(drift);
  });

  const glintCount = context.compact ? 70 : 180;
  const glintPositions = new Float32Array(glintCount * 3);
  const random = seededRandom(4417);
  for (let index = 0; index < glintCount; index += 1) {
    const x = (random() - 0.5) * 84;
    const z = (random() - 0.5) * 84;
    glintPositions[index * 3] = x;
    glintPositions[index * 3 + 1] = polarGroundHeight(x, z, false) + 0.05;
    glintPositions[index * 3 + 2] = z;
  }
  const glintGeometry = new THREE.BufferGeometry();
  glintGeometry.setAttribute('position', new THREE.BufferAttribute(glintPositions, 3));
  const glintMaterial = new THREE.PointsMaterial({
    color: '#ffffff',
    size: 0.09,
    map: context.glow,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  context.world.add(new THREE.Points(glintGeometry, glintMaterial));
  context.update.push((time) => {
    glintMaterial.opacity = 0.45 + Math.sin(time * 0.7) * 0.1;
  });
}

/** Deterministic, irregular outline for the meltwater lead — no straight edges. */
function leadEdgeFactor(theta: number): number {
  return (
    1 +
    Math.sin(theta * 3 + 0.7) * 0.14 +
    Math.sin(theta * 5 - 1.2) * 0.09 +
    Math.sin(theta * 2 + 1.9) * 0.05
  );
}

/** Radial, ringed surface so the water can ripple across the whole organic shape. */
function createLeadGeometry(compact: boolean): THREE.BufferGeometry {
  const segments = compact ? 32 : 56;
  const rings = compact ? 5 : 8;
  const radiusX = 21;
  const radiusZ = 4.7;
  const positions: number[] = [0, 0, 0];
  const factors = Array.from({ length: segments }, (_, index) =>
    leadEdgeFactor((index / segments) * Math.PI * 2),
  );

  for (let ring = 1; ring <= rings; ring += 1) {
    const t = ring / rings;
    for (let segment = 0; segment < segments; segment += 1) {
      const theta = (segment / segments) * Math.PI * 2;
      const factor = factors[segment];
      positions.push(
        Math.cos(theta) * radiusX * factor * t,
        0,
        (Math.sin(theta) * radiusZ * factor + Math.sin(theta * 2) * 0.45) * t,
      );
    }
  }

  const indices: number[] = [];
  for (let segment = 0; segment < segments; segment += 1) {
    const next = (segment + 1) % segments;
    indices.push(0, 1 + next, 1 + segment);
  }
  for (let ring = 1; ring < rings; ring += 1) {
    const inner = 1 + (ring - 1) * segments;
    const outer = 1 + ring * segments;
    for (let segment = 0; segment < segments; segment += 1) {
      const next = (segment + 1) % segments;
      indices.push(inner + segment, outer + segment, inner + next);
      indices.push(outer + segment, outer + next, inner + next);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** An irregular meltwater lead with broken ice at its edges and a soft glitter. */
function addAntarcticDayLead(context: WorldContext): void {
  const waterY = 0.22;
  const waterGeometry = createLeadGeometry(context.compact);
  const waterMaterial = new THREE.MeshStandardMaterial({
    color: '#2f7898',
    roughness: 0.28,
    metalness: 0.12,
    transparent: true,
    opacity: 0.94,
    side: THREE.DoubleSide,
  });
  const water = new THREE.Mesh(waterGeometry, waterMaterial);
  water.position.set(0, waterY, -20);
  water.receiveShadow = !context.compact;
  context.world.add(water);

  const base = new Float32Array(waterGeometry.attributes.position.array);
  context.update.push((time) => {
    const position = waterGeometry.attributes.position as THREE.BufferAttribute;
    for (let index = 0; index < position.count; index += 1) {
      const x = base[index * 3];
      const z = base[index * 3 + 2];
      position.setY(
        index,
        Math.sin(x * 0.55 + time * 1.3) * 0.022 + Math.sin(z * 0.85 + time * 0.9) * 0.015,
      );
    }
    position.needsUpdate = true;
    waterGeometry.computeVertexNormals();
  });

  const floeMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.78,
    metalness: 0.02,
    flatShading: true,
  });
  const random = seededRandom(8823);
  for (let index = 0; index < (context.compact ? 12 : 22); index += 1) {
    const theta = random() * Math.PI * 2;
    const factor = leadEdgeFactor(theta);
    const x = Math.cos(theta) * 21 * factor + (random() - 0.5) * 0.8;
    const z =
      -20 + (Math.sin(theta) * 4.7 * factor + Math.sin(theta * 2) * 0.45) + (random() - 0.5) * 0.8;
    const size = 0.4 + random() * 0.75;
    const floe = new THREE.Mesh(
      organicGeometry(1, 1, index * 4.7 + 1.3, '#84acc8', '#f4fbff', 0.22),
      floeMaterial,
    );
    floe.position.set(x, waterY + 0.04, z);
    floe.scale.set(size * 1.4, size * 0.32, size);
    floe.rotation.y = random() * Math.PI;
    context.world.add(floe);
  }

  const sparkleCount = context.compact ? 36 : 90;
  const positions = new Float32Array(sparkleCount * 3);
  for (let index = 0; index < sparkleCount; index += 1) {
    const theta = random() * Math.PI * 2;
    const t = Math.sqrt(random());
    const factor = leadEdgeFactor(theta);
    positions[index * 3] = Math.cos(theta) * 21 * factor * t;
    positions[index * 3 + 1] = waterY + 0.06;
    positions[index * 3 + 2] =
      -20 + (Math.sin(theta) * 4.7 * factor + Math.sin(theta * 2) * 0.45) * t;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: '#d9f4ff',
    size: 0.12,
    map: context.glow,
    transparent: true,
    opacity: 0.6,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  context.world.add(new THREE.Points(geometry, material));
  context.update.push((time) => {
    material.opacity = 0.48 + Math.sin(time * 2.2) * 0.16;
  });
}

/** Rounded, stratified icebergs rather than flat tabular slabs. */
function addAntarcticDayIcebergs(context: WorldContext): void {
  const bergs = [
    { x: -17, z: -19, s: 1.35, seed: 1.4 },
    { x: 13, z: -23, s: 2.05, seed: 2.9 },
    { x: -23, z: -30, s: 1.7, seed: 4.6 },
    { x: 21, z: -31, s: 1.45, seed: 6.2 },
    { x: -6, z: -34, s: 1.15, seed: 7.9 },
    { x: 27, z: -14, s: 1.05, seed: 9.4 },
  ].map(({ x, z, s, seed }) => {
    const group = new THREE.Group();
    const body = new THREE.Mesh(
      organicGeometry(1, 1, seed, '#6f9dbd', '#f4fbff', 0.24),
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.7,
        metalness: 0.04,
        flatShading: true,
      }),
    );
    body.scale.set(s, s * 0.86, s);
    const waterline = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 10),
      new THREE.MeshStandardMaterial({ color: '#5d8dae', roughness: 0.5, transparent: true, opacity: 0.55 }),
    );
    waterline.scale.set(s * 1.06, s * 0.16, s * 1.02);
    waterline.position.y = -s * 0.42;
    group.add(body, waterline);
    group.position.set(x, polarGroundHeight(x, z, false) + s * 0.32, z);
    context.world.add(group);
    return { group, baseY: group.position.y, seed };
  });

  context.update.push((time) => {
    bergs.forEach(({ group, baseY, seed }) => {
      group.position.y = baseY + Math.sin(time * 0.55 + seed) * 0.1;
      group.rotation.y = Math.sin(time * 0.11 + seed) * 0.14;
    });
  });
}

/** Emperor adult or downy chick with more anatomical detail than the night penguin. */
function createEmperorPenguin(context: WorldContext, chick = false): THREE.Group {
  const group = new THREE.Group();
  const dark = standard(chick ? '#6b7681' : '#1a2c40', 0.9);
  const pale = standard(chick ? '#c9d1d8' : '#f1eedf', 0.86);
  const gold = standard('#e4b65a', 0.72);
  const orange = standard('#d9822b', 0.7);
  const ink = standard('#0d1620', 0.4);
  const glint = standard('#ffffff', 0.3);

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.5, 18, 14), dark);
  body.scale.set(0.9, 1.55, 0.78);
  body.position.y = 0.95;
  const belly = new THREE.Mesh(new THREE.SphereGeometry(0.42, 18, 14), pale);
  belly.scale.set(0.78, 1.18, 0.32);
  belly.position.set(0, 0.92, 0.34);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.3, 18, 14), dark);
  head.position.y = 1.72;
  const face = new THREE.Mesh(new THREE.SphereGeometry(0.21, 16, 12), pale);
  face.scale.set(0.8, 0.86, 0.32);
  face.position.set(0, 1.68, 0.24);

  group.add(body, belly, head, face);
  const headParts: THREE.Object3D[] = [head, face];
  if (!chick) {
    [-1, 1].forEach((side) => {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 10), gold);
      ear.scale.set(0.55, 1.35, 0.45);
      ear.position.set(side * 0.27, 1.62, 0.04);
      group.add(ear);
      headParts.push(ear);
    });
    const bib = new THREE.Mesh(new THREE.SphereGeometry(0.2, 14, 10), gold);
    bib.scale.set(0.72, 0.42, 0.18);
    bib.position.set(0, 1.37, 0.4);
    group.add(bib);
    headParts.push(bib);
  }

  const bill = new THREE.Mesh(new THREE.ConeGeometry(0.085, 0.3, 10), orange);
  bill.rotation.x = Math.PI / 2;
  bill.position.set(0, 1.62, 0.42);
  group.add(bill);
  headParts.push(bill);
  [-1, 1].forEach((side) => {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.038, 10, 8), ink);
    eye.scale.set(1, 1.1, 0.55);
    eye.position.set(side * 0.14, 1.78, 0.29);
    const catchlight = new THREE.Mesh(new THREE.SphereGeometry(0.011, 8, 6), glint);
    catchlight.position.set(side * 0.14 - 0.008, 1.79, 0.315);
    group.add(eye, catchlight);
    headParts.push(eye, catchlight);
  });

  const flippers = [-1, 1].map((side) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.45, 1.3, 0);
    const flipper = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 10), dark);
    flipper.scale.set(0.62, 2.7, 0.42);
    flipper.position.y = -0.5;
    pivot.add(flipper);
    pivot.rotation.z = side * 0.22;
    group.add(pivot);
    return pivot;
  });
  const feet = [-1, 1].map((side) => {
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), orange);
    foot.scale.set(1.05, 0.3, 1.5);
    foot.position.set(side * 0.2, 0.05, 0.18);
    group.add(foot);
    return foot;
  });

  group.traverse((child) => {
    if (child instanceof THREE.Mesh && !context.compact) child.castShadow = true;
  });
  group.userData.flippers = flippers;
  group.userData.feet = feet;
  group.userData.headParts = headParts.map((object) => ({ object, y: object.position.y }));
  group.userData.body = { mesh: body, scaleY: body.scale.y };
  group.scale.setScalar(chick ? 0.62 : 1);
  context.world.add(group);
  return group;
}

/** Huddles of adults and chicks, plus a loose walking line across the ice. */
function addEmperorColony(context: WorldContext): void {
  const random = seededRandom(20261127);
  const clusters = [
    { x: -8, z: -9, radius: 4.2, adults: context.compact ? 3 : 5, chicks: context.compact ? 1 : 3 },
    { x: 3, z: -13, radius: 4.8, adults: 6, chicks: 3 },
    { x: 12, z: -8, radius: 3.6, adults: 4, chicks: 2 },
    { x: -1, z: -20, radius: 4.4, adults: 5, chicks: 2 },
    { x: 19, z: -18, radius: 3.2, adults: context.compact ? 2 : 3, chicks: 1 },
  ];
  const huddled: Array<{ penguin: THREE.Group; baseY: number; phase: number; resting: boolean }> = [];

  clusters.forEach((cluster) => {
    const place = (chick: boolean, count: number) => {
      for (let index = 0; index < count; index += 1) {
        const angle = random() * Math.PI * 2;
        const radius = Math.sqrt(random()) * cluster.radius;
        const x = cluster.x + Math.cos(angle) * radius;
        const z = cluster.z + Math.sin(angle) * radius;
        const penguin = createEmperorPenguin(context, chick);
        const scale = (chick ? 0.4 : 0.56) + random() * 0.12;
        penguin.scale.setScalar(scale);
        const resting = !chick && index % 3 === 0;
        if (resting) penguin.scale.y *= 0.82;
        penguin.rotation.y = angle + Math.PI * 0.5 + (random() - 0.5) * 0.8;
        const baseY = polarGroundHeight(x, z, false);
        penguin.position.set(x, baseY, z);
        huddled.push({ penguin, baseY, phase: random() * Math.PI * 2, resting });
      }
    };
    place(false, cluster.adults);
    place(true, cluster.chicks);
  });

  const walkers = Array.from({ length: context.compact ? 6 : 10 }, (_, index) => {
    const penguin = createEmperorPenguin(context, false);
    penguin.scale.setScalar(0.55 + random() * 0.11);
    penguin.rotation.y = Math.PI / 2;
    const z = -4 + index * 1.05 + (random() - 0.5) * 0.6;
    penguin.position.set(-38 + index * 7.2, polarGroundHeight(-38, z, false), z);
    return { penguin, z, phase: random() * Math.PI * 2, speed: 0.46 + random() * 0.12 };
  });

  context.update.push((time) => {
    const animateDetails = (penguin: THREE.Group, phase: number, walking: boolean) => {
      const headParts = penguin.userData.headParts as
        | Array<{ object: THREE.Object3D; y: number }>
        | undefined;
      headParts?.forEach(({ object, y }) => {
        object.position.y =
          y + Math.sin(time * (walking ? 2.4 : 1.35) + phase) * (walking ? 0.045 : 0.03);
      });
      const body = penguin.userData.body as { mesh: THREE.Mesh; scaleY: number } | undefined;
      if (body) {
        body.mesh.scale.y = body.scaleY * (1 + Math.sin(time * 1.1 + phase) * 0.012);
      }
    };

    huddled.forEach(({ penguin, baseY, phase, resting }) => {
      const sway = Math.sin(time * 0.75 + phase) * (resting ? 0.014 : 0.026);
      penguin.position.y = baseY + Math.abs(sway);
      penguin.rotation.z = sway * 0.5;
      const flippers = penguin.userData.flippers as THREE.Group[] | undefined;
      flippers?.forEach((flipper, index) => {
        const side = index === 0 ? -1 : 1;
        const preen = resting ? Math.sin(time * 1.1 + phase) * 0.18 : 0;
        flipper.rotation.z = side * (0.22 + Math.abs(sway) * 3 + preen);
      });
      animateDetails(penguin, phase, false);
    });

    walkers.forEach(({ penguin, z, phase, speed }) => {
      const x = -38 + ((time * speed + phase * 2.2) % 76);
      const step = time * 4.2 + phase;
      penguin.position.x = x;
      penguin.position.z = z + Math.sin(time * 0.28 + phase) * 0.35;
      penguin.position.y = polarGroundHeight(penguin.position.x, penguin.position.z, false) + Math.abs(Math.sin(step)) * 0.04;
      penguin.rotation.z = Math.sin(step) * 0.045;
      const flippers = penguin.userData.flippers as THREE.Group[] | undefined;
      flippers?.forEach((flipper, index) => {
        const side = index === 0 ? -1 : 1;
        flipper.rotation.z = side * (0.2 + Math.abs(Math.sin(step)) * 0.22);
      });
      animateDetails(penguin, phase, true);
    });
  });
}

/** A Weddell seal with a spotted coat, whiskers, and slow breathing. */
function addAntarcticDaySeal(context: WorldContext): THREE.Group {
  const group = new THREE.Group();
  const fur = standard('#837b72', 0.94);
  const pale = standard('#b8b0a5', 0.94);
  const ink = standard('#12100f', 0.35);
  const spot = standard('#5f5a54', 1);

  const body = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), fur);
  body.scale.set(2.1, 0.72, 0.78);
  body.position.y = 0.72;
  const belly = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), pale);
  belly.scale.set(1.8, 0.44, 0.63);
  belly.position.set(0, 0.53, 0.14);
  const neck = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), fur);
  neck.scale.set(0.78, 0.68, 0.62);
  neck.position.set(1.65, 0.98, 0);
  const head = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), fur);
  head.scale.set(0.62, 0.5, 0.48);
  head.position.set(2.25, 1.08, 0);
  const snout = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), pale);
  snout.scale.set(0.27, 0.22, 0.3);
  snout.position.set(2.72, 0.96, 0);
  group.add(body, belly, neck, head, snout);

  const foreFlippers: THREE.Mesh[] = [];
  const hindFlippers: THREE.Mesh[] = [];
  [-1, 1].forEach((side) => {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.065, 10, 8), ink);
    eye.position.set(2.42, 1.22, side * 0.34);
    const nostril = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), ink);
    nostril.position.set(2.9, 1.03, side * 0.11);
    group.add(eye, nostril);

    for (let index = 0; index < 4; index += 1) {
      const points = [
        new THREE.Vector3(2.78, 0.94, side * 0.2),
        new THREE.Vector3(3.05, 0.88 - index * 0.035, side * (0.55 + index * 0.08)),
      ];
      group.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(points),
          new THREE.LineBasicMaterial({ color: '#d8d0c4', transparent: true, opacity: 0.75 }),
        ),
      );
    }

    const foreFlipper = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), fur);
    foreFlipper.scale.set(0.75, 0.12, 0.28);
    foreFlipper.position.set(0.7, 0.3, side * 0.72);
    foreFlipper.rotation.y = side * 0.42;
    group.add(foreFlipper);
    foreFlippers.push(foreFlipper);

    const hindFlipper = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), fur);
    hindFlipper.scale.set(0.55, 0.1, 0.34);
    hindFlipper.position.set(-2.0, 0.26, side * 0.46);
    hindFlipper.rotation.y = -side * 0.3;
    group.add(hindFlipper);
    hindFlippers.push(hindFlipper);
  });

  const random = seededRandom(3371);
  for (let index = 0; index < (context.compact ? 10 : 18); index += 1) {
    const x = -1.6 + random() * 3.2;
    const side = random() > 0.5 ? 1 : -1;
    const z = side * (0.35 + random() * 0.36);
    const y = 0.72 + Math.sin((x + 1.6) / 3.2 * Math.PI) * 0.55;
    const mark = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), spot);
    mark.scale.set(1 + random() * 0.7, 0.35, 0.8);
    mark.position.set(x, y, z);
    group.add(mark);
  }

  group.userData.head = head;
  group.userData.snout = snout;
  group.userData.foreFlippers = foreFlippers;
  group.userData.hindFlippers = hindFlippers;
  group.scale.setScalar(0.82);
  group.position.set(-6, polarGroundHeight(-6, 7, false) + 0.02, 7);
  group.rotation.y = 0.55;
  context.world.add(group);
  context.update.push((time) => {
    group.scale.y = 0.82 * (1 + Math.sin(time * 1.5) * 0.03);
    const lift = Math.sin(time * 0.85);
    const head = group.userData.head as THREE.Mesh | undefined;
    if (head) {
      head.position.y = 1.08 + lift * 0.05;
      head.rotation.y = Math.sin(time * 0.42) * 0.12;
    }
    const snout = group.userData.snout as THREE.Mesh | undefined;
    if (snout) snout.position.y = 0.96 + lift * 0.045;
    const foreFlippers = group.userData.foreFlippers as THREE.Mesh[] | undefined;
    foreFlippers?.forEach((flipper, index) => {
      flipper.rotation.x = Math.sin(time * 1.1 + index * Math.PI) * 0.12;
    });
    const hindFlippers = group.userData.hindFlippers as THREE.Mesh[] | undefined;
    hindFlippers?.forEach((flipper, index) => {
      flipper.rotation.x = Math.sin(time * 0.9 + index * Math.PI) * 0.07;
    });
  });
  return group;
}

/** An orca surfacing in the lead, with saddle patch, eye patches, and flukes. */
function addAntarcticDayOrca(context: WorldContext): THREE.Group {
  const group = new THREE.Group();
  const black = standard('#101821', 0.4);
  const white = standard('#f4f4ef', 0.5);
  const ink = standard('#05080d', 0.3);

  const body = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), black);
  body.scale.set(3.0, 0.85, 0.8);
  const belly = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), white);
  belly.scale.set(2.5, 0.44, 0.62);
  belly.position.set(0.2, -0.36, 0);
  const saddle = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 10), white);
  saddle.scale.set(0.75, 0.16, 0.36);
  saddle.position.set(-0.7, 0.78, 0);
  group.add(body, belly, saddle);

  const fins: THREE.Mesh[] = [];
  [-1, 1].forEach((side) => {
    const patch = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), white);
    patch.scale.set(0.34, 0.17, 0.1);
    patch.position.set(1.35, 0.24, side * 0.72);
    patch.rotation.y = side * 0.3;
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), ink);
    eye.position.set(1.42, 0.25, side * 0.79);
    const fin = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), black);
    fin.scale.set(0.82, 0.1, 0.3);
    fin.position.set(0.5, -0.42, side * 0.78);
    fin.rotation.y = side * 0.5;
    group.add(patch, eye, fin);
    fins.push(fin);
  });

  const dorsal = new THREE.Mesh(new THREE.ConeGeometry(0.46, 1.6, 12), black);
  dorsal.position.set(0.1, 1.18, 0);
  dorsal.rotation.z = 0.28;
  const tailStock = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), black);
  tailStock.scale.set(0.8, 0.28, 0.3);
  tailStock.position.x = -3.0;
  group.add(dorsal, tailStock);
  const flukes: THREE.Mesh[] = [];
  [-1, 1].forEach((side) => {
    const fluke = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), black);
    fluke.scale.set(0.55, 0.08, 0.85);
    fluke.position.set(-3.55, 0, side * 0.65);
    fluke.rotation.y = side * 0.35;
    group.add(fluke);
    flukes.push(fluke);
  });

  group.userData.fins = fins;
  group.userData.flukes = flukes;
  group.scale.setScalar(0.78);
  context.world.add(group);
  context.update.push((time) => {
    const swim = time * 0.17;
    group.position.set(
      Math.cos(swim) * 8,
      0.08 + Math.sin(time * 0.8) * 0.13,
      -20 + Math.sin(swim) * 2.6,
    );
    group.rotation.y = Math.atan2(-Math.cos(swim) * 2.6, -Math.sin(swim) * 8);
    group.rotation.z = Math.cos(time * 0.8) * 0.09;
    const flukes = group.userData.flukes as THREE.Mesh[] | undefined;
    flukes?.forEach((fluke, index) => {
      fluke.rotation.x = Math.sin(time * 3.4 + index * 0.4) * 0.22;
    });
    const fins = group.userData.fins as THREE.Mesh[] | undefined;
    fins?.forEach((fin, index) => {
      fin.rotation.x = Math.sin(time * 1.8 + index * Math.PI) * 0.08;
    });
  });
  return group;
}

/** Sparse horizontal streaks of wind-blown snow, not falling flakes. */
function addAntarcticDaySpindrift(context: WorldContext): void {
  const count = context.compact ? 70 : 160;
  const positions = new Float32Array(count * 6);
  const origins: Array<[number, number, number, number]> = [];
  for (let index = 0; index < count; index += 1) {
    origins.push([rand(-46, 46), rand(0.25, 3.4), rand(-40, 26), 0.3 + rand(0, 0.7)]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: '#ffffff',
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
  });
  context.world.add(new THREE.LineSegments(geometry, material));
  context.update.push((time, delta) => {
    origins.forEach((origin, index) => {
      origin[0] += delta * (1.3 + (index % 5) * 0.22);
      if (origin[0] > 46) origin[0] = -46;
      const x = origin[0];
      const y = origin[1] + Math.sin(time * 0.9 + index) * 0.09;
      const z = origin[2];
      positions.set([x, y, z, x - origin[3], y + 0.02, z], index * 6);
    });
    geometry.attributes.position.needsUpdate = true;
  });
}

/**
 * A slow, silent orbit around the fast-ice colony. The camera never drops
 * through the terrain, and the full ring of mountains keeps every angle full.
 */
function addAntarcticDayOrbit(context: WorldContext, camera: THREE.PerspectiveCamera): void {
  const target = new THREE.Vector3(0, 4.8, -10);
  const orbit = { theta: 0.12, phi: 1.36, radius: 34 };
  const orbitGoal = { ...orbit };
  const placeCamera = () => {
    camera.position.set(
      target.x + orbit.radius * Math.sin(orbit.phi) * Math.sin(orbit.theta),
      target.y + orbit.radius * Math.cos(orbit.phi),
      target.z + orbit.radius * Math.sin(orbit.phi) * Math.cos(orbit.theta),
    );
    camera.position.y = Math.max(camera.position.y, polarGroundHeight(camera.position.x, camera.position.z, false) + 2.8);
    camera.lookAt(target);
  };
  placeCamera();

  context.update.push((_time, delta) => {
    orbitGoal.theta += delta * 0.016;
    const smoothing = Math.min(1, delta * 2.4);
    orbit.theta += (orbitGoal.theta - orbit.theta) * smoothing;
    orbit.phi += (orbitGoal.phi - orbit.phi) * smoothing;
    orbit.radius += (orbitGoal.radius - orbit.radius) * smoothing;
    placeCamera();
  });
}

/**
 * Daylit fast ice built from the ground up: ringed nunataks, a sculpted ice
 * sheet, a penguin crèche, a Weddell seal, an orca in a meltwater lead, and
 * wind-blown spindrift. Drawn only from real-time geometry and shaders.
 */
function buildAntarcticDay(context: WorldContext, camera: THREE.PerspectiveCamera): void {
  addAntarcticDaySky(context);
  addAntarcticDayMountains(context);
  addAntarcticDayTerrain(context);
  addLights(context, {
    sky: '#e6f2fb',
    ground: '#b7d2e4',
    sun: '#fff7e8',
    sunPower: 2.2,
    ambient: 0.68,
    sunPosition: DAY_LIGHT_POSITION,
  });
  addAntarcticDayLead(context);
  addAntarcticDayIcebergs(context);
  addAntarcticDaySeal(context);
  addAntarcticDayOrca(context);
  addEmperorColony(context);
  addAntarcticDaySpindrift(context);
  addAntarcticDayOrbit(context, camera);
}

function buildWinter(context: WorldContext, night: boolean, camera: THREE.PerspectiveCamera): void {
  if (!night) {
    buildAntarcticDay(context, camera);
    return;
  }
  addAuroraSky(context);
  addAntarcticBackdrop(context, true);
  addPolarSnowfall(context, context.compact ? 320 : 900);
  addLights(context, {
    sky: '#7fb3df',
    ground: '#3f6587',
    sun: '#b8d2ff',
    sunPower: 0.38,
    ambient: 0.28,
  });
  // Flying birds stay in the night sky; daylight belongs to penguins, seals, and orca.
  addAntarcticPetrels(context, true);
  addAurora(context);
  addMoon(context, [-30, 18, -60], 1.1);
  addGlow(context, '#dff1ff', [-30, 18, -60], 18, 0.7);
  addWater(context, { color: '#061c2e', z: 16, width: 46, depth: 10, wave: 0.045 });
  addAntarcticWildlife(context, true);
  addPenguinWalkers(context, true);
}

/* -------------------------------------------------------------------------
   Summer night: a port of the desert-after-dark reference. The scene is made
   only from real-time geometry and shaders: dune field, buttes, cacti, moon,
   campfire, and the animals that come out after sunset. There are no sprites
   standing in for scenery and no drawn text, labels, or controls.
   ------------------------------------------------------------------------- */

const DESERT_CAMP_X = -52;
const DESERT_CAMP_Z = -70;

function desertHash(x: number, z: number): number {
  const value = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return value - Math.floor(value);
}

function desertNoise(x: number, z: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  return (
    desertHash(ix, iz) * (1 - ux) * (1 - uz) +
    desertHash(ix + 1, iz) * ux * (1 - uz) +
    desertHash(ix, iz + 1) * (1 - ux) * uz +
    desertHash(ix + 1, iz + 1) * ux * uz
  );
}

function desertFbm(x: number, z: number): number {
  let sum = 0;
  let amplitude = 0.5;
  for (let index = 0; index < 4; index += 1) {
    sum += amplitude * desertNoise(x, z);
    x = x * 2.03 + 11;
    z = z * 2.03 + 7;
    amplitude *= 0.5;
  }
  return sum;
}

function desertSmooth(a: number, b: number, x: number): number {
  const clamped = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return clamped * clamped * (3 - 2 * clamped);
}

function desertHeight(x: number, z: number): number {
  const radius = Math.hypot(x, z);
  const blend = desertSmooth(16, 70, radius);
  const ridge = x * 0.045 + z * 0.02 + desertFbm(x * 0.015, z * 0.015) * 4;
  const dune =
    Math.pow(1 - Math.abs(Math.sin(ridge)), 1.6) *
    (0.5 + 0.9 * desertFbm(x * 0.02 + 9, z * 0.02));
  const campDistance =
    ((x - DESERT_CAMP_X) ** 2 + (z - DESERT_CAMP_Z) ** 2) / 500;
  return (
    blend * (2 + 13 * dune + 2 * desertFbm(x * 0.05, z * 0.05)) +
    (1 - blend) * 0.5 * desertFbm(x * 0.08, z * 0.08) +
    7 * Math.exp(-campDistance)
  );
}

function desertGlowTexture(context: WorldContext): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const brush = canvas.getContext('2d');
  if (brush) {
    const gradient = brush.createRadialGradient(64, 64, 0, 64, 64, 64);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.15, 'rgba(235,240,255,0.8)');
    gradient.addColorStop(0.4, 'rgba(160,190,255,0.2)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    brush.fillStyle = gradient;
    brush.fillRect(0, 0, 128, 128);
  }
  const texture = new THREE.CanvasTexture(canvas);
  context.textures.push(texture);
  return texture;
}

function desertMoonTexture(context: WorldContext): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const brush = canvas.getContext('2d');
  if (brush) {
    brush.fillStyle = '#ebe9e2';
    brush.fillRect(0, 0, 256, 256);
    for (let index = 0; index < 12; index += 1) {
      const x = Math.random() * 256;
      const y = Math.random() * 256;
      const radius = 20 + Math.random() * 40;
      const crater = brush.createRadialGradient(x, y, 0, x, y, radius);
      crater.addColorStop(0, 'rgba(120,125,135,0.5)');
      crater.addColorStop(1, 'rgba(120,125,135,0)');
      brush.fillStyle = crater;
      brush.fillRect(0, 0, 256, 256);
    }
    for (let index = 0; index < 50; index += 1) {
      brush.strokeStyle = 'rgba(150,150,150,0.3)';
      brush.beginPath();
      brush.arc(Math.random() * 256, Math.random() * 256, 2 + Math.random() * 7, 0, 7);
      brush.stroke();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  context.textures.push(texture);
  return texture;
}

function desertPart(
  parent: THREE.Object3D,
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  scale: [number, number, number],
  position: [number, number, number],
  rotation?: [number, number, number],
): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.scale.set(...scale);
  mesh.position.set(...position);
  if (rotation) mesh.rotation.set(...rotation);
  mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}

function desertAt<T extends THREE.Object3D>(
  context: WorldContext,
  object: T,
  x: number,
  z: number,
  y = 0,
): T {
  object.position.set(x, desertHeight(x, z) + y, z);
  context.world.add(object);
  return object;
}

interface DesertScorpion {
  group: THREE.Group;
  tail: THREE.Group;
  origin: [number, number];
  phase: number;
}

interface DesertBat {
  group: THREE.Group;
  leftWing: THREE.Group;
  rightWing: THREE.Group;
}

interface DesertTumbleweed {
  mesh: THREE.Mesh;
  offset: number;
  z: number;
}

function buildDesertNight(context: WorldContext, camera: THREE.PerspectiveCamera): void {
  const timeUniform = { value: 0 };
  const blacklightUniform = { value: 0 };
  const glow = desertGlowTexture(context);

  context.world.fog = new THREE.FogExp2(0x141a30, 0.0034);
  const fogDay = new THREE.Color(0x141a30);

  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(1500, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: { t: timeUniform, k: blacklightUniform },
      vertexShader: `
        varying vec3 v;
        void main() {
          v = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 v;
        uniform float t;
        uniform float k;
        float h(vec2 p) {
          p = fract(p * vec2(123.34, 456.21));
          p += dot(p, p + 45.32);
          return fract(p.x * p.y);
        }
        float n(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(h(i), h(i + vec2(1.0, 0.0)), f.x),
            mix(h(i + vec2(0.0, 1.0)), h(i + vec2(1.0, 1.0)), f.x),
            f.y
          );
        }
        float fb(vec2 p) {
          return 0.5 * n(p) + 0.25 * n(p * 2.03 + 7.0) +
            0.125 * n(p * 4.1 + 3.0) + 0.06 * n(p * 8.3);
        }
        void main() {
          vec3 d = normalize(v);
          float y = d.y;
          vec3 c = mix(vec3(0.05, 0.07, 0.14), vec3(0.004, 0.008, 0.025), pow(max(y, 0.0), 0.5));
          c += vec3(0.16, 0.08, 0.05) * exp(-abs(y) * 9.0) * 0.5;
          for (int i = 0; i < 2; i++) {
            float s = 170.0 - float(i) * 90.0;
            vec3 p = d * s;
            vec3 id = floor(p);
            float hh = fract(sin(dot(id, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
            float b = step(0.986 + 0.009 * float(i), hh) *
              smoothstep(0.35, 0.0, length(fract(p) - 0.5));
            c += mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.85, 0.65), fract(hh * 97.0)) *
              b * (0.55 + 0.45 * sin(t * 3.0 + hh * 600.0)) *
              (1.1 + float(i)) * smoothstep(-0.02, 0.1, y);
          }
          vec3 nn = normalize(vec3(0.3, 0.8, 0.5));
          vec3 t1 = normalize(cross(nn, vec3(0.0, 1.0, 0.0)));
          vec3 t2 = cross(nn, t1);
          float bb = dot(d, nn);
          float a = atan(dot(d, t2), dot(d, t1));
          float mw = exp(-bb * bb * 14.0) * (0.35 + 0.9 * fb(vec2(a * 4.0, bb * 14.0))) *
            (1.0 - 0.6 * smoothstep(0.55, 0.8, fb(vec2(a * 7.0 + 3.0, bb * 30.0))));
          c += mix(vec3(0.55, 0.62, 0.9), vec3(0.95, 0.8, 0.6), fb(vec2(a * 3.0, bb * 8.0))) *
            mw * 0.5 * smoothstep(-0.05, 0.15, y);
          c = mix(c, c * vec3(0.6, 0.3, 1.2) + vec3(0.015, 0.0, 0.04), k);
          gl_FragColor = vec4(c, 1.0);
        }
      `,
    }),
  );
  sky.renderOrder = -1;
  context.world.add(sky);

  const hemi = new THREE.HemisphereLight(0x3a5fa0, 0x1a1408, 0.5);
  context.world.add(hemi);
  const moonDirection = new THREE.Vector3(-0.45, 0.27, -0.85).normalize();
  const moonLight = new THREE.DirectionalLight(0xb4cfff, 1.5);
  moonLight.position.copy(moonDirection).multiplyScalar(300);
  moonLight.castShadow = true;
  moonLight.shadow.mapSize.set(context.compact ? 1024 : 2048, context.compact ? 1024 : 2048);
  moonLight.shadow.bias = -0.0004;
  moonLight.shadow.normalBias = 0.05;
  const moonShadow = moonLight.shadow.camera;
  moonShadow.left = -60;
  moonShadow.bottom = -60;
  moonShadow.right = 60;
  moonShadow.top = 60;
  moonShadow.near = 10;
  moonShadow.far = 700;
  moonShadow.updateProjectionMatrix();
  context.world.add(moonLight);
  const ultraviolet = new THREE.DirectionalLight(0x7a3cff, 0);
  ultraviolet.position.set(20, 100, 30);
  context.world.add(ultraviolet);
  const fireLight = new THREE.PointLight(0xff8a3a, 2.4, 45, 1.5);
  fireLight.position.set(0, 2.2, 0);
  context.world.add(fireLight);

  const moon = new THREE.Mesh(
    new THREE.SphereGeometry(40, 32, 16),
    new THREE.MeshBasicMaterial({
      map: desertMoonTexture(context),
      fog: false,
      toneMapped: false,
    }),
  );
  moon.position.copy(moonDirection).multiplyScalar(1100);
  context.world.add(moon);
  const moonGlow = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: glow,
      color: 0xaac4ff,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      fog: false,
      opacity: 0.8,
    }),
  );
  moonGlow.scale.set(460, 460, 1);
  moonGlow.position.copy(moon.position);
  context.world.add(moonGlow);

  const shootingStar = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(0, 0, -70),
    ]),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, fog: false }),
  );
  const shootingState = { time: 9, velocity: new THREE.Vector3() };
  context.world.add(shootingStar);

  const terrainSegments = context.compact ? 150 : 260;
  const terrainGeometry = new THREE.PlaneGeometry(
    800,
    800,
    terrainSegments,
    terrainSegments,
  );
  terrainGeometry.rotateX(-Math.PI / 2);
  const terrainPosition = terrainGeometry.attributes.position as THREE.BufferAttribute;
  for (let index = 0; index < terrainPosition.count; index += 1) {
    terrainPosition.setY(
      index,
      desertHeight(terrainPosition.getX(index), terrainPosition.getZ(index)),
    );
  }
  terrainGeometry.computeVertexNormals();
  const terrainNormal = terrainGeometry.attributes.normal as THREE.BufferAttribute;
  const terrainColors = new Float32Array(terrainPosition.count * 3);
  for (let index = 0; index < terrainPosition.count; index += 1) {
    const shadeAmount =
      0.7 + 0.4 * Math.min(1, terrainNormal.getY(index));
    const variation = desertFbm(terrainPosition.getX(index) * 0.25, terrainPosition.getZ(index) * 0.25);
    terrainColors[index * 3] = (0.66 + 0.2 * variation) * shadeAmount;
    terrainColors[index * 3 + 1] = (0.55 + 0.17 * variation) * shadeAmount;
    terrainColors[index * 3 + 2] = (0.4 + 0.12 * variation) * shadeAmount;
  }
  terrainGeometry.setAttribute('color', new THREE.BufferAttribute(terrainColors, 3));

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = 256;
  bumpCanvas.height = 256;
  const bumpBrush = bumpCanvas.getContext('2d');
  if (bumpBrush) {
    bumpBrush.fillStyle = '#808080';
    bumpBrush.fillRect(0, 0, 256, 256);
    for (let index = 0; index < 90; index += 1) {
      const y = Math.random() * 256;
      bumpBrush.strokeStyle = index % 2 ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.18)';
      bumpBrush.lineWidth = 1 + Math.random() * 2;
      bumpBrush.beginPath();
      for (let x = 0; x <= 256; x += 8) {
        bumpBrush.lineTo(x, y + Math.sin(x * 0.05 + index) * 5 + Math.sin(x * 0.13 + index * 2) * 2);
      }
      bumpBrush.stroke();
    }
  }
  const bumpTexture = new THREE.CanvasTexture(bumpCanvas);
  bumpTexture.wrapS = THREE.RepeatWrapping;
  bumpTexture.wrapT = THREE.RepeatWrapping;
  bumpTexture.repeat.set(60, 60);
  context.textures.push(bumpTexture);
  const terrain = new THREE.Mesh(
    terrainGeometry,
    new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.95,
      bumpMap: bumpTexture,
      bumpScale: 1.4,
    }),
  );
  terrain.receiveShadow = true;
  terrain.castShadow = true;
  context.world.add(terrain);

  const sphereGeometry = new THREE.SphereGeometry(1, 16, 12);
  const coneGeometry = new THREE.ConeGeometry(1, 1, 10);
  const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 8);
  const standardMaterial = (color: number, roughness = 0.85) =>
    new THREE.MeshStandardMaterial({ color, roughness });
  const basicMaterial = (color: number) => new THREE.MeshBasicMaterial({ color });
  const foxCoat = standardMaterial(0xe0bd8c);
  const foxCream = standardMaterial(0xf5e8d2);
  const darkFur = standardMaterial(0x251c14, 0.5);
  const pinkEar = standardMaterial(0xd9928a);
  const rockMaterial = new THREE.MeshStandardMaterial({
    color: 0x5c5148,
    roughness: 1,
    flatShading: true,
  });

  for (let index = 0; index < 7; index += 1) {
    const angle = index * 0.9 + 0.3;
    const distance = 300 + Math.random() * 60;
    const heightValue = 50 + Math.random() * 50;
    const geometry = new THREE.CylinderGeometry(
      14 + Math.random() * 10,
      26 + Math.random() * 14,
      heightValue,
      9,
      5,
    );
    const position = geometry.attributes.position as THREE.BufferAttribute;
    for (let vertex = 0; vertex < position.count; vertex += 1) {
      const factor =
        0.85 +
        desertHash(position.getX(vertex) * 0.3 + index, position.getZ(vertex) * 0.3) * 0.3;
      position.setX(vertex, position.getX(vertex) * factor);
      position.setZ(vertex, position.getZ(vertex) * factor);
    }
    geometry.computeVertexNormals();
    const butte = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0x5b3a2a,
        roughness: 1,
        flatShading: true,
      }),
    );
    butte.position.set(
      Math.cos(angle) * distance,
      heightValue / 2 - 8,
      Math.sin(angle) * distance,
    );
    context.world.add(butte);
  }

  const addRock = (x: number, z: number, scale: number) => {
    const geometry = new THREE.IcosahedronGeometry(1, 1);
    const position = geometry.attributes.position as THREE.BufferAttribute;
    for (let vertex = 0; vertex < position.count; vertex += 1) {
      const factor =
        0.75 + desertHash(position.getX(vertex) * 9 + x, position.getZ(vertex) * 7 + z) * 0.5;
      position.setXYZ(
        vertex,
        position.getX(vertex) * factor,
        position.getY(vertex) * factor * 0.7,
        position.getZ(vertex) * factor,
      );
    }
    geometry.computeVertexNormals();
    const rock = new THREE.Mesh(geometry, rockMaterial);
    rock.scale.setScalar(scale);
    rock.castShadow = true;
    rock.receiveShadow = true;
    rock.rotation.y = Math.random() * 6;
    return desertAt(context, rock, x, z, scale * 0.25);
  };
  for (let index = 0; index < 26; index += 1) {
    const angle = Math.random() * Math.PI * 2;
    const distance = 8 + Math.random() * 70;
    addRock(Math.cos(angle) * distance, Math.sin(angle) * distance, 0.5 + Math.random() * 2);
  }

  const cactusMaterial = standardMaterial(0x2d5a37, 0.85);
  const ribbedCylinder = (length: number, radius: number) => {
    const geometry = new THREE.CylinderGeometry(radius * 0.92, radius, length, 16, 6);
    const position = geometry.attributes.position as THREE.BufferAttribute;
    for (let vertex = 0; vertex < position.count; vertex += 1) {
      const x = position.getX(vertex);
      const z = position.getZ(vertex);
      const factor = 1 + 0.06 * Math.cos(Math.atan2(z, x) * 12);
      position.setX(vertex, x * factor);
      position.setZ(vertex, z * factor);
    }
    geometry.computeVertexNormals();
    return geometry;
  };
  const cactusCap = (radius: number) => new THREE.SphereGeometry(radius, 14, 8);
  for (let index = 0; index < 13; index += 1) {
    const angle = Math.random() * Math.PI * 2;
    const distance = 16 + Math.random() * 75;
    const x = Math.cos(angle) * distance;
    const z = Math.sin(angle) * distance;
    const cactus = new THREE.Group();
    const addCactusPart = (
      geometry: THREE.BufferGeometry,
      px: number,
      py: number,
      pz: number,
      rz = 0,
    ) => {
      const mesh = new THREE.Mesh(geometry, cactusMaterial);
      mesh.position.set(px, py, pz);
      mesh.rotation.z = rz;
      mesh.castShadow = true;
      cactus.add(mesh);
      return mesh;
    };
    addCactusPart(ribbedCylinder(8, 0.6), 0, 4, 0);
    addCactusPart(cactusCap(0.56), 0, 8, 0).scale.y = 0.8;
    ([[-1, 4.2, 2.6], [1, 3.2, 3.6]] as const).forEach(([side, armHeight, armLength]) => {
      addCactusPart(
        ribbedCylinder(1.8, 0.36),
        side * 1.1,
        armHeight,
        0,
        Math.PI / 2,
      );
      addCactusPart(cactusCap(0.36), side * 2, armHeight, 0);
      addCactusPart(ribbedCylinder(armLength, 0.34), side * 2, armHeight + armLength / 2, 0);
      addCactusPart(cactusCap(0.34), side * 2, armHeight + armLength, 0).scale.y = 0.8;
    });
    cactus.scale.setScalar(0.8 + Math.random() * 0.8);
    cactus.rotation.y = Math.random() * 6;
    desertAt(context, cactus, x, z, -0.3);
  }

  const flameMaterial = (color: number, opacity: number) =>
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
  const fire = new THREE.Group();
  fire.position.y = desertHeight(0, 0);
  context.world.add(fire);
  const flames = ([
    [0.7, 2.2, 0xff6a12, 0.8],
    [0.5, 1.7, 0xffa028, 0.8],
    [0.3, 1.2, 0xffe080, 0.9],
  ] as const).map(([radius, heightValue, color, opacity], index) => {
    const mesh = new THREE.Mesh(
      new THREE.ConeGeometry(radius, heightValue, 10),
      flameMaterial(color, opacity),
    );
    mesh.position.y = heightValue / 2 + 0.3;
    fire.add(mesh);
    return { mesh, height: heightValue, phase: index * 2 };
  });
  for (let index = 0; index < 3; index += 1) {
    desertPart(
      fire,
      cylinderGeometry,
      standardMaterial(0x3a2416),
      [0.2, 2.4, 0.2],
      [0, 0.3, 0],
      [0, index * 1.05, Math.PI / 2],
    );
  }
  for (let index = 0; index < 9; index += 1) {
    const angle = index * 0.7;
    desertPart(
      fire,
      sphereGeometry,
      rockMaterial,
      [0.5, 0.35, 0.5],
      [Math.cos(angle) * 1.7, 0.2, Math.sin(angle) * 1.7],
    );
  }
  const fireGlow = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: glow,
      color: 0xff8a2a,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.9,
    }),
  );
  fireGlow.scale.set(10, 10, 1);
  fireGlow.position.y = 1.6;
  fire.add(fireGlow);
  const sparkPositions = new Float32Array(180);
  for (let index = 0; index < sparkPositions.length; index += 3) {
    sparkPositions[index] = (Math.random() - 0.5) * 1.2;
    sparkPositions[index + 1] = Math.random() * 9;
    sparkPositions[index + 2] = (Math.random() - 0.5) * 1.2;
  }
  const sparkGeometry = new THREE.BufferGeometry();
  sparkGeometry.setAttribute('position', new THREE.BufferAttribute(sparkPositions, 3));
  const sparks = new THREE.Points(
    sparkGeometry,
    new THREE.PointsMaterial({
      size: 0.4,
      map: glow,
      color: 0xffa040,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  sparks.frustumCulled = false;
  fire.add(sparks);

  const fox = new THREE.Group();
  desertPart(fox, sphereGeometry, foxCoat, [0.45, 0.5, 0.95], [0, 0.9, 0]);
  desertPart(fox, sphereGeometry, foxCream, [0.38, 0.4, 0.6], [0, 0.8, 0.25]);
  desertPart(fox, sphereGeometry, foxCoat, [0.36, 0.33, 0.4], [0, 1.15, 0.95]);
  desertPart(fox, sphereGeometry, foxCream, [0.16, 0.14, 0.3], [0, 1.06, 1.28]);
  desertPart(fox, sphereGeometry, darkFur, [0.07, 0.06, 0.06], [0, 1.08, 1.56]);
  const foxEars = [-1, 1].map((side) => {
    const ear = new THREE.Group();
    ear.position.set(side * 0.25, 1.4, 0.85);
    desertPart(ear, coneGeometry, foxCoat, [0.2, 1, 0.1], [0, 0.5, 0]);
    desertPart(ear, coneGeometry, pinkEar, [0.13, 0.8, 0.06], [0, 0.45, 0.03]);
    ear.rotation.z = -side * 0.22;
    fox.add(ear);
    return ear;
  });
  [-1, 1].forEach((side) =>
    desertPart(
      fox,
      sphereGeometry,
      basicMaterial(0xffc060),
      [0.06, 0.06, 0.06],
      [side * 0.15, 1.22, 1.22],
    ),
  );
  const foxLegs = ([[-1, 0.4], [1, 0.4], [-1, -0.4], [1, -0.4]] as const).map(
    ([side, z]) => {
      const leg = new THREE.Group();
      leg.position.set(side * 0.22, 0.75, z);
      desertPart(leg, cylinderGeometry, foxCoat, [0.07, 0.8, 0.07], [0, -0.4, 0]);
      desertPart(leg, sphereGeometry, darkFur, [0.08, 0.05, 0.11], [0, -0.8, 0.03]);
      fox.add(leg);
      return leg;
    },
  );
  desertPart(fox, sphereGeometry, foxCoat, [0.28, 0.28, 0.9], [0, 0.85, -1.05], [0.5, 0, 0]);
  desertPart(fox, sphereGeometry, darkFur, [0.22, 0.22, 0.3], [0, 0.62, -1.65]);
  fox.scale.setScalar(1.25);
  context.world.add(fox);

  const scorpionShell = new THREE.MeshStandardMaterial({
    color: 0x3a2a1a,
    roughness: 0.5,
    emissive: new THREE.Color(0x000000),
  });
  const makeScorpion = (): { group: THREE.Group; tail: THREE.Group } => {
    const group = new THREE.Group();
    desertPart(group, sphereGeometry, scorpionShell, [0.5, 0.18, 0.62], [0, 0.25, 0]);
    desertPart(group, sphereGeometry, scorpionShell, [0.4, 0.16, 0.55], [0, 0.22, -0.7]);
    const tail = new THREE.Group();
    group.add(tail);
    ([
      [0.3, -1], [0.55, -1.35], [0.95, -1.55], [1.4, -1.5],
      [1.75, -1.2], [1.9, -0.75], [1.85, -0.3],
    ] as const).forEach(([y, z], index) => {
      const radius = 0.17 - index * 0.012;
      desertPart(tail, sphereGeometry, scorpionShell, [radius, radius, radius], [0, y, z]);
    });
    desertPart(tail, coneGeometry, scorpionShell, [0.07, 0.3, 0.07], [0, 1.76, -0.2], [2.21, 0, 0]);
    [-1, 1].forEach((side) => {
      desertPart(
        group,
        sphereGeometry,
        scorpionShell,
        [0.1, 0.1, 0.45],
        [side * 0.4, 0.25, 0.85],
        [0, -side * 0.35, 0],
      );
      desertPart(group, sphereGeometry, scorpionShell, [0.24, 0.13, 0.38], [side * 0.62, 0.25, 1.3]);
      desertPart(group, sphereGeometry, basicMaterial(0x66ffcc), [0.04, 0.04, 0.04], [side * 0.1, 0.38, 0.45]);
      [0.35, 0.12, -0.12, -0.35].forEach((z) =>
        desertPart(
          group,
          cylinderGeometry,
          scorpionShell,
          [0.035, 0.55, 0.035],
          [side * 0.5, 0.14, z],
          [0, 0, side * 0.95],
        ),
      );
    });
    group.scale.setScalar(2.3);
    return { group, tail };
  };
  const scorpions: DesertScorpion[] = ([
    [5.5, 5.5], [-7, 5], [9, -4],
  ] as const).map(([x, z], index) => {
    const { group, tail } = makeScorpion();
    context.world.add(desertAt(context, group, x, z));
    addRock(x + 1.8, z - 1.5, 0.8);
    return { group, tail, origin: [x, z], phase: index * 2 };
  });

  const snakeSand = standardMaterial(0xb59a6a, 0.7);
  const snakeBands = standardMaterial(0x8a7048, 0.7);
  const snakeSegments: THREE.Object3D[] = [];
  let snakeHead = new THREE.Group();
  for (let index = 0; index < 30; index += 1) {
    if (index === 0) {
      snakeHead = new THREE.Group();
      desertPart(snakeHead, sphereGeometry, snakeSand, [0.46, 0.4, 0.58], [0, 0, 0]);
      [-1, 1].forEach((side) =>
        desertPart(
          snakeHead,
          sphereGeometry,
          basicMaterial(0xffd040),
          [0.07, 0.07, 0.07],
          [side * 0.2, 0.14, 0.3],
        ),
      );
      context.world.add(snakeHead);
      snakeSegments.push(snakeHead);
    } else {
      const radius = 0.36 * (1 - index / 46) + 0.06;
      const segment = new THREE.Mesh(sphereGeometry, index % 2 ? snakeSand : snakeBands);
      segment.scale.set(radius, radius * 0.85, radius);
      segment.castShadow = true;
      context.world.add(segment);
      snakeSegments.push(segment);
    }
  }

  const owl = new THREE.Group();
  const owlBody = standardMaterial(0xf1ead8, 0.9);
  const owlFace = standardMaterial(0xfbf6ea, 0.9);
  const owlFeather = standardMaterial(0xb9946a, 0.9);
  desertPart(owl, sphereGeometry, owlBody, [0.4, 0.38, 0.9], [0, 0, 0]);
  desertPart(owl, sphereGeometry, owlBody, [0.36, 0.34, 0.34], [0, 0.12, 0.95]);
  desertPart(owl, sphereGeometry, owlFace, [0.3, 0.3, 0.08], [0, 0.12, 1.22]);
  [-1, 1].forEach((side) =>
    desertPart(owl, sphereGeometry, basicMaterial(0x050505), [0.07, 0.07, 0.07], [side * 0.11, 0.16, 1.28]),
  );
  desertPart(owl, coneGeometry, standardMaterial(0xd9b36a), [0.05, 0.18, 0.05], [0, 0.05, 1.34], [Math.PI / 2, 0, 0]);
  desertPart(owl, sphereGeometry, owlBody, [0.25, 0.04, 0.5], [0, 0, -1.1]);
  const owlWings = [-1, 1].map((side) => {
    const wing = new THREE.Group();
    wing.position.set(side * 0.3, 0.1, 0.2);
    desertPart(wing, sphereGeometry, owlBody, [1.8, 0.05, 0.75], [side * 1.8, 0, 0]);
    desertPart(wing, sphereGeometry, owlFeather, [1, 0.05, 0.6], [side * 2.9, 0, 0.1]);
    owl.add(wing);
    return wing;
  });
  owl.scale.setScalar(1.3);
  context.world.add(owl);

  const wingShape = new THREE.Shape();
  ([
    [0, 0.1], [0.55, 0.28], [1.05, 0.04], [0.88, -0.14],
    [0.62, -0.04], [0.46, -0.3], [0.2, -0.24], [0, -0.22],
  ] as const).forEach(([x, y], index) => {
    if (index === 0) wingShape.moveTo(x, y);
    else wingShape.lineTo(x, y);
  });
  const wingGeometryRight = new THREE.ShapeGeometry(wingShape).rotateX(Math.PI / 2);
  const wingGeometryLeft = new THREE.ShapeGeometry(wingShape).rotateX(Math.PI / 2).scale(-1, 1, 1);
  const batMaterial = new THREE.MeshStandardMaterial({
    color: 0x14100e,
    roughness: 1,
    side: THREE.DoubleSide,
  });
  const bats: DesertBat[] = Array.from({ length: 9 }, () => {
    const group = new THREE.Group();
    desertPart(group, sphereGeometry, batMaterial, [0.12, 0.1, 0.2], [0, 0, 0]);
    desertPart(group, sphereGeometry, batMaterial, [0.08, 0.08, 0.08], [0, 0.02, 0.22]);
    [-1, 1].forEach((side) =>
      desertPart(group, coneGeometry, batMaterial, [0.03, 0.1, 0.02], [side * 0.04, 0.1, 0.24]),
    );
    const rightWing = new THREE.Group();
    const leftWing = new THREE.Group();
    rightWing.add(new THREE.Mesh(wingGeometryRight, batMaterial));
    leftWing.add(new THREE.Mesh(wingGeometryLeft, batMaterial));
    group.add(rightWing, leftWing);
    group.scale.setScalar(1.3);
    context.world.add(group);
    return { group, leftWing, rightWing };
  });

  const coyote = new THREE.Group();
  const coyoteFur = standardMaterial(0x2b2520, 1);
  desertPart(coyote, sphereGeometry, coyoteFur, [0.42, 0.5, 1], [0, 1, 0]);
  desertPart(coyote, sphereGeometry, coyoteFur, [0.34, 0.45, 0.5], [0, 1.05, 0.7]);
  desertPart(coyote, cylinderGeometry, coyoteFur, [0.2, 0.9, 0.2], [0, 1.6, 0.95], [0.65, 0, 0]);
  desertPart(coyote, sphereGeometry, coyoteFur, [0.27, 0.25, 0.36], [0, 2.05, 1.35]);
  desertPart(coyote, coneGeometry, coyoteFur, [0.13, 0.6, 0.13], [0, 2.3, 1.65], [0.95, 0, 0]);
  [-1, 1].forEach((side) => {
    desertPart(coyote, coneGeometry, coyoteFur, [0.1, 0.35, 0.06], [side * 0.14, 2.3, 1.2], [-0.1, 0, -side * 0.15]);
    desertPart(coyote, sphereGeometry, basicMaterial(0xffe070), [0.04, 0.04, 0.04], [side * 0.16, 2.12, 1.5]);
  });
  ([[-1, 0.55], [1, 0.55], [-1, -0.55], [1, -0.55]] as const).forEach(([side, z]) =>
    desertPart(coyote, cylinderGeometry, coyoteFur, [0.08, 1, 0.08], [side * 0.2, 0.5, z]),
  );
  desertPart(coyote, sphereGeometry, coyoteFur, [0.13, 0.13, 0.7], [0, 0.95, -1.15], [-0.5, 0, 0]);
  coyote.scale.setScalar(3.4);
  coyote.rotation.y = -Math.PI / 2;
  desertAt(context, coyote, DESERT_CAMP_X, DESERT_CAMP_Z);

  const tumbleweeds: DesertTumbleweed[] = [0, 1].map((index) => {
    const mesh = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.1, 1),
      new THREE.MeshStandardMaterial({ color: 0x7a5d38, wireframe: true }),
    );
    mesh.castShadow = true;
    context.world.add(mesh);
    return { mesh, offset: index * 60, z: -14 + index * 34 };
  });

  const orbitTarget = new THREE.Vector3(0, 10, 0);
  const orbit = { theta: 0.15, phi: 1.45, radius: 45 };
  const orbitGoal = { ...orbit };
  const placeCamera = () => {
    camera.position.set(
      orbitTarget.x + orbit.radius * Math.sin(orbit.phi) * Math.sin(orbit.theta),
      orbitTarget.y + orbit.radius * Math.cos(orbit.phi),
      orbitTarget.z + orbit.radius * Math.sin(orbit.phi) * Math.cos(orbit.theta),
    );
    camera.position.y = Math.max(
      camera.position.y,
      desertHeight(camera.position.x, camera.position.z) + 2.5,
    );
    camera.lookAt(orbitTarget);
  };
  placeCamera();

  context.update.push((time, delta) => {
    timeUniform.value = time;
    blacklightUniform.value = 0;
    moonLight.intensity = 1.5;
    hemi.intensity = 0.5;
    ultraviolet.intensity = 0;
    if (context.world.fog instanceof THREE.FogExp2) {
      context.world.fog.color.copy(fogDay);
    }
    scorpionShell.emissive.setRGB(0, 0, 0);

    fireLight.intensity =
      2.4 * (0.85 + 0.15 * Math.sin(time * 23) * Math.sin(time * 17 + 1) + 0.1);
    flames.forEach(({ mesh, height, phase }) => {
      const flicker = 0.85 + 0.25 * Math.sin(time * 9 + phase) * Math.sin(time * 13 + phase * 2);
      mesh.scale.set(
        1 + 0.15 * Math.sin(time * 11 + phase),
        flicker,
        1 + 0.15 * Math.cos(time * 10 + phase),
      );
      mesh.position.y = (height * flicker) / 2 + 0.3;
    });
    fireGlow.material.opacity = 0.75 + 0.2 * Math.sin(time * 14);
    for (let index = 0; index < sparkPositions.length; index += 3) {
      sparkPositions[index + 1] += delta * (2 + (index % 7) * 0.4);
      sparkPositions[index] += Math.sin(time * 2 + index) * delta * 0.8;
      if (sparkPositions[index + 1] > 10) {
        sparkPositions[index] = (Math.random() - 0.5) * 1.2;
        sparkPositions[index + 1] = 0.8;
        sparkPositions[index + 2] = (Math.random() - 0.5) * 1.2;
      }
    }
    sparkGeometry.attributes.position.needsUpdate = true;

    const foxAngle = time * 0.35;
    const foxRadius = 9 + Math.sin(time * 0.5) * 2;
    const foxX = Math.cos(foxAngle) * foxRadius;
    const foxZ = Math.sin(foxAngle) * foxRadius;
    const foxStep = Math.sin(time * 9);
    fox.position.set(foxX, desertHeight(foxX, foxZ) + Math.abs(foxStep) * 0.06, foxZ);
    fox.rotation.y = Math.atan2(-Math.sin(foxAngle), Math.cos(foxAngle));
    foxLegs.forEach((leg, index) => {
      leg.rotation.x =
        Math.sin(time * 9 + (index % 2 ? Math.PI : 0) + (index > 1 ? Math.PI / 2 : 0)) * 0.6;
    });
    const earTwitch = Math.pow(Math.max(0, Math.sin(time * 0.7)), 8) * 0.4;
    foxEars.forEach((ear, index) => {
      ear.rotation.z = (index ? 1 : -1) * (0.22 - earTwitch * Math.sin(time * 30));
    });

    scorpions.forEach(({ group, tail, origin, phase }) => {
      const [x, z] = origin;
      const angle = time * 0.15 + phase;
      const px = x + Math.cos(angle) * 1.2;
      const pz = z + Math.sin(angle) * 1.2;
      group.position.set(px, desertHeight(px, pz), pz);
      group.rotation.y = Math.atan2(-Math.sin(angle), Math.cos(angle));
      tail.rotation.y = Math.sin(time * 1.4 + phase) * 0.18;
    });

    snakeSegments.forEach((segment, index) => {
      const snakeTime = time - index * 0.15;
      const angle = snakeTime * 0.3;
      const lateral = Math.sin(snakeTime * 4) * 0.9;
      const distance = 7 + lateral;
      const x = 13 + Math.cos(angle) * distance;
      const z = -9 + Math.sin(angle) * distance;
      segment.position.set(x, desertHeight(x, z) + 0.2, z);
      if (index === 0) {
        snakeHead.rotation.y =
          Math.atan2(-Math.sin(angle), Math.cos(angle)) + Math.cos(snakeTime * 4) * 0.5;
      }
    });

    const owlAngle = time * 0.12;
    owl.position.set(
      Math.cos(owlAngle) * 34,
      17 + Math.sin(time * 0.5) * 2,
      Math.sin(owlAngle) * 34,
    );
    owl.rotation.y = Math.atan2(-Math.sin(owlAngle), Math.cos(owlAngle));
    owl.rotation.z = 0.25;
    const owlFlap = Math.sin(time * 3) * 0.35;
    owlWings.forEach((wing, index) => {
      wing.rotation.z = (index ? 1 : -1) * owlFlap;
    });

    const batPoint = (angle: number, index: number): [number, number, number] => [
      Math.cos(angle) * (10 + (index % 4) * 3) + Math.sin(angle * 0.7) * 4,
      5 + (index % 3) * 1.5 + Math.sin(angle * 2.1) * 2.5,
      Math.sin(angle * 1.3) * (10 + (index % 3) * 3),
    ];
    bats.forEach(({ group, leftWing, rightWing }, index) => {
      const angle = time * (0.5 + index * 0.04) + index * 1.9;
      const point = batPoint(angle, index);
      const next = batPoint(angle + 0.05, index);
      const flap = Math.sin(time * 20 + index) * 0.8;
      group.position.set(...point);
      group.rotation.y = Math.atan2(next[0] - point[0], next[2] - point[2]);
      rightWing.rotation.z = flap;
      leftWing.rotation.z = -flap;
    });

    coyote.rotation.z = Math.sin(time * 0.8) * 0.02;

    tumbleweeds.forEach(({ mesh, offset, z }) => {
      const x = ((time * 3 + offset) % 140) - 70;
      mesh.position.set(
        x,
        desertHeight(x, z) + 1.1 + Math.abs(Math.sin(time * 2 + offset)) * 0.5,
        z,
      );
      mesh.rotation.z = -x * 0.9;
      mesh.rotation.x = time;
    });

    shootingState.time += delta;
    if (shootingState.time > 1.1 && Math.random() < delta * 0.12) {
      shootingState.time = 0;
      const azimuth = Math.random() * Math.PI * 2;
      const elevation = 0.35 + Math.random() * 0.8;
      shootingStar.position.set(
        Math.cos(azimuth) * Math.cos(elevation) * 1000,
        Math.sin(elevation) * 1000,
        Math.sin(azimuth) * Math.cos(elevation) * 1000,
      );
      shootingState.velocity
        .set(Math.random() - 0.5, -0.3 - Math.random() * 0.3, Math.random() - 0.5)
        .normalize()
        .multiplyScalar(500);
      const end = shootingStar.position.clone().add(shootingState.velocity);
      shootingStar.lookAt(end);
    }
    if (shootingState.time <= 1) {
      shootingStar.position.addScaledVector(shootingState.velocity, delta);
      (shootingStar.material as THREE.LineBasicMaterial).opacity =
        Math.sin(Math.PI * shootingState.time) * 0.9;
    } else {
      (shootingStar.material as THREE.LineBasicMaterial).opacity = 0;
    }

    // Unlike the reference there is no toggle and no written control: the
    // desert always keeps orbiting whenever motion is allowed.
    orbitGoal.theta += delta * 0.035;
    const smoothing = Math.min(1, delta * 3);
    orbit.theta += (orbitGoal.theta - orbit.theta) * smoothing;
    orbit.phi += (orbitGoal.phi - orbit.phi) * smoothing;
    orbit.radius += (orbitGoal.radius - orbit.radius) * smoothing;
    placeCamera();
  });
}

function buildSummer(context: WorldContext, night: boolean): void {
  addSky(context, night
    ? [[0, '#120f2d'], [0.56, '#342350'], [1, '#8b5064']]
    : [[0, '#7fc8e6'], [0.54, '#d9f2f4'], [1, '#fff1cd']]);
  addWater(context, {
    color: night ? '#3a3a72' : '#7fc8dd',
    z: -19,
    wave: night ? 0.09 : 0.16,
  });
  addDuneGround(context, night);
  addShoreFoam(context, night);
  addBeachDetails(context, night);
  addLights(context, {
    sky: night ? '#c5b6e8' : '#eaf7ff',
    ground: night ? '#241e3c' : '#e8c88f',
    sun: night ? '#c5b6ff' : '#fff1cf',
    sunPower: night ? 0.42 : 2.5,
    ambient: night ? 0.32 : 0.75,
  });

  if (night) {
    addStars(context, context.compact ? 90 : 260, '#fff0c9');
    addMoon(context, [8.5, 12.5, -30], 1.7);
    addFireflies(context, context.compact ? 24 : 80, '#ffe6a3');
    addGlow(context, '#ffe6aa', [8.5, 0.05, -8], 12, 0.24);
  } else {
    addFireflies(context, context.compact ? 16 : 44, '#fff4c4');
    addGulls(context, context.compact ? 3 : 6);
    addGlow(context, '#ffd78a', [-8, 12, -26], 26, 0.32);
  }

  const palm = createPalm(context);
  palm.position.set(-6.2, 0, 0.8);
  palm.scale.setScalar(night ? 0.9 : 1.05);
  const parasol = night ? null : createParasol(context, '#f7e9cc', '#d87845');
  if (parasol) {
    parasol.position.set(4.8, 0, 1.4);
    parasol.scale.setScalar(0.9);
  }
  const boat = createSailboat(context, night);
  boat.position.set(0.6, 0, -7.5);
  boat.scale.setScalar(0.95);
  const crab = night ? null : createCrab(context);
  if (crab) crab.position.set(-2.4, 0, 3.4);

  const lanterns: THREE.Group[] = [];
  if (night) {
    ['#ffd98a', '#f2a6c0', '#9fd8ff'].forEach((color, index) => {
      const lantern = createLantern(context, color);
      lantern.position.set(-3.5 + index * 3.2, 1.5 + (index % 2) * 0.4, -2.5 - index * 1.2);
      lanterns.push(lantern);
    });
  }

  context.update.push((time) => {
    const crown = palm.userData.crown as THREE.Group | undefined;
    if (crown) crown.rotation.z = Math.sin(time * 0.7) * 0.05;
    boat.position.y = Math.sin(time * 1.1) * 0.1;
    boat.rotation.z = Math.sin(time * 0.9) * 0.06;
    if (parasol) parasol.rotation.z = Math.sin(time * 0.8) * 0.02;
    if (crab) crab.position.x = -2.4 + Math.sin(time * 0.9) * 0.45;
    lanterns.forEach((lantern, index) => {
      lantern.position.y = 1.5 + Math.sin(time * 0.8 + index) * 0.3;
      lantern.position.x += Math.sin(time * 0.3 + index) * 0.002;
    });
  });
}

/* -------------------------------------------------------------------------
   Rainforest rain season. The reference is a wet Amazon river seen through
   canopy: winding water, mist, light shafts, a bow after the storm, and small
   animals going about their day. Everything below is real-time WebGL geometry.
   ------------------------------------------------------------------------- */

const RAINFOREST_TERRAIN_Z = -14;

function rainforestRiverX(z: number): number {
  return Math.sin(z * 0.085) * 4.1 + Math.sin(z * 0.031 + 1.2) * 2.8;
}

function rainforestHeight(x: number, z: number): number {
  const distance = x - rainforestRiverX(z);
  const channel =
    Math.exp(-(distance * distance) / 22) * Math.max(0, 1 - Math.pow(Math.abs(z) / 74, 4));
  const hills =
    0.72 +
    Math.sin(x * 0.09) * Math.cos(z * 0.075) * 1.1 +
    Math.sin(x * 0.21 + z * 0.13) * 0.42 +
    Math.pow(Math.hypot(x, z) / 66, 2) * 4.2;
  return hills * (1 - channel) - 1.05 * channel;
}

function rainforestGroundTexture(context: WorldContext): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const paint = canvas.getContext('2d');
  if (paint) {
    paint.fillStyle = '#c8c8bd';
    paint.fillRect(0, 0, 128, 128);
    const random = seededRandom(8123);
    for (let index = 0; index < 420; index += 1) {
      paint.fillStyle = random() < 0.52 ? 'rgba(22,48,24,0.14)' : 'rgba(255,255,240,0.12)';
      paint.beginPath();
      paint.ellipse(
        random() * 128,
        random() * 128,
        1 + random() * 5,
        0.6 + random() * 2,
        random() * Math.PI,
        0,
        Math.PI * 2,
      );
      paint.fill();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(26, 26);
  texture.colorSpace = THREE.SRGBColorSpace;
  context.textures.push(texture);
  return texture;
}

function rainforestShaftTexture(context: WorldContext): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const paint = canvas.getContext('2d');
  if (paint) {
    const gradient = paint.createLinearGradient(0, 0, 0, 64);
    gradient.addColorStop(0, 'rgba(255,255,255,0)');
    gradient.addColorStop(0.38, 'rgba(255,255,255,0.16)');
    gradient.addColorStop(0.52, 'rgba(255,255,255,0.9)');
    gradient.addColorStop(0.68, 'rgba(255,255,255,0.14)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    paint.fillStyle = gradient;
    paint.fillRect(0, 0, 64, 64);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  context.textures.push(texture);
  return texture;
}

function addRainforestTerrain(context: WorldContext, night: boolean): void {
  const segments = context.compact ? 76 : 132;
  const geometry = new THREE.PlaneGeometry(78, 78, segments, segments);
  geometry.rotateX(-Math.PI / 2);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const texture = rainforestGroundTexture(context);
  const shadow = new THREE.Color(night ? '#09170f' : '#33542f');
  const moss = new THREE.Color(night ? '#1d4b30' : '#6f9b55');
  const soil = new THREE.Color(night ? '#101f15' : '#6c5d3f');
  const colors: number[] = [];

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const z = position.getZ(index) + RAINFOREST_TERRAIN_Z;
    const height = rainforestHeight(x, z);
    position.setY(index, height);

    const bankDistance = Math.abs(x - rainforestRiverX(z));
    const wet = Math.exp(-(bankDistance * bankDistance) / 46);
    const shade = THREE.MathUtils.clamp(
      0.35 + Math.sin(x * 0.19 + z * 0.14) * 0.18 + Math.sin(z * 0.6) * 0.08 + height * 0.03,
      0,
      1,
    );
    const color = soil.clone().lerp(shadow, 0.55).lerp(moss, shade * 0.72).lerp(shadow, wet * 0.42);
    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  position.needsUpdate = true;
  geometry.computeVertexNormals();

  const terrain = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      map: texture,
      vertexColors: true,
      roughness: 0.98,
      side: THREE.DoubleSide,
    }),
  );
  terrain.position.z = RAINFOREST_TERRAIN_Z;
  terrain.receiveShadow = !context.compact;
  context.world.add(terrain);
}

function addRainforestRiver(context: WorldContext, night: boolean): void {
  const steps = context.compact ? 56 : 108;
  const cross = context.compact ? 5 : 9;
  const positions: number[] = [];
  const indices: number[] = [];

  for (let row = 0; row <= steps; row += 1) {
    const z = -53 + (row / steps) * 78;
    const center = rainforestRiverX(z);
    const width = 4.7 + Math.sin(z * 0.11) * 0.6;
    for (let column = 0; column <= cross; column += 1) {
      const t = (column / cross) * 2 - 1;
      positions.push(center + t * width, -0.82 + Math.sin(z * 0.18) * 0.04, z);
    }
  }
  for (let row = 0; row < steps; row += 1) {
    for (let column = 0; column < cross; column += 1) {
      const a = row * (cross + 1) + column;
      const b = a + 1;
      const c = a + cross + 1;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const water = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: night ? '#1e4e49' : '#3f9b90',
      roughness: 0.22,
      metalness: 0.14,
      transparent: true,
      opacity: 0.94,
      side: THREE.DoubleSide,
    }),
  );
  water.receiveShadow = !context.compact;
  context.world.add(water);

  const base = new Float32Array(positions);
  context.update.push((time) => {
    const attribute = geometry.attributes.position as THREE.BufferAttribute;
    for (let index = 0; index < attribute.count; index += 1) {
      const x = base[index * 3];
      const z = base[index * 3 + 2];
      attribute.setY(
        index,
        base[index * 3 + 1] +
          Math.sin(x * 0.65 + time * 1.25) * 0.075 +
          Math.sin(z * 0.8 + time * 0.85) * 0.045,
      );
    }
    attribute.needsUpdate = true;
    geometry.computeVertexNormals();
  });

  const rippleCount = context.compact ? 42 : 96;
  const ripplePositions = new Float32Array(rippleCount * 6);
  const random = seededRandom(991);
  const origins: Array<[number, number]> = [];
  for (let index = 0; index < rippleCount; index += 1) {
    origins.push([-50 + random() * 70, (random() * 2 - 1) * 3.6]);
  }
  const rippleGeometry = new THREE.BufferGeometry();
  rippleGeometry.setAttribute('position', new THREE.BufferAttribute(ripplePositions, 3));
  const ripples = new THREE.LineSegments(
    rippleGeometry,
    new THREE.LineBasicMaterial({
      color: night ? '#cfe8ff' : '#f4fff2',
      transparent: true,
      opacity: night ? 0.3 : 0.26,
      depthWrite: false,
    }),
  );
  ripples.frustumCulled = false;
  context.world.add(ripples);
  context.update.push((time) => {
    origins.forEach(([originZ, offset], index) => {
      const z = (((originZ + time * 0.9 + 53) % 78) + 78) % 78 - 53;
      const x = rainforestRiverX(z) + offset;
      const length = 0.35 + (index % 4) * 0.12;
      ripplePositions.set([x, -0.74, z, x + length, -0.74, z], index * 6);
    });
    rippleGeometry.attributes.position.needsUpdate = true;
  });
}

function addRainforestLightShafts(context: WorldContext, night: boolean): void {
  const texture = rainforestShaftTexture(context);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    color: night ? '#bfe8ff' : '#fff0b8',
    transparent: true,
    opacity: night ? 0.065 : 0.13,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
  });
  const direction = new THREE.Vector3(-0.45, 0.7, 0.55).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const random = seededRandom(night ? 523 : 271);
  const count = context.compact ? 4 : 8;

  for (let index = 0; index < count; index += 1) {
    const beam = new THREE.Mesh(
      new THREE.CylinderGeometry(1.1 + random() * 1.1, 3.6 + random() * 2.8, 66, 12, 1, true),
      material,
    );
    beam.quaternion.setFromUnitVectors(up, direction);
    beam.position
      .set((random() - 0.5) * 34, 4, -32 + random() * 34)
      .addScaledVector(direction, 30);
    context.world.add(beam);
  }
}

function addRainforestMist(context: WorldContext, night: boolean): void {
  const random = seededRandom(613);
  const count = context.compact ? 3 : 6;
  for (let index = 0; index < count; index += 1) {
    const z = -42 + index * 11 + random() * 3;
    const x = rainforestRiverX(z) + (random() - 0.5) * 6;
    addGlow(
      context,
      night ? '#9fd8b7' : '#f2fff0',
      [x, 0.2 + random() * 0.7, z],
      12 + random() * 8,
      night ? 0.1 : 0.12,
    );
  }
}

function addRainforestBow(context: WorldContext, night: boolean): void {
  const arcs = [
    { inner: 15, outer: 22, invert: 0, opacity: night ? 0.15 : 0.26 },
    { inner: 23, outer: 31, invert: 1, opacity: night ? 0.08 : 0.12 },
  ];

  arcs.forEach((arc) => {
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uInner: { value: arc.inner },
        uOuter: { value: arc.outer },
        uInvert: { value: arc.invert },
        uOpacity: { value: arc.opacity },
        uNight: { value: night ? 1 : 0 },
      },
      vertexShader: `
        varying vec3 vPosition;
        void main() {
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uInner;
        uniform float uOuter;
        uniform float uInvert;
        uniform float uOpacity;
        uniform float uNight;
        varying vec3 vPosition;

        vec3 spectrum(float h) {
          return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
        }

        void main() {
          float t = clamp((length(vPosition.xy) - uInner) / (uOuter - uInner), 0.0, 1.0);
          if (uInvert > 0.5) t = 1.0 - t;
          float alpha =
            pow(sin(3.14159 * t), 0.8) *
            smoothstep(0.0, 0.35, vPosition.y / uOuter) *
            uOpacity;
          vec3 color = spectrum((1.0 - t) * 0.78);
          if (uNight > 0.5) color = mix(color, vec3(0.72, 0.94, 1.0), 0.45);
          gl_FragColor = vec4(color * alpha, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: false,
    });
    material.toneMapped = false;
    const bow = new THREE.Mesh(new THREE.RingGeometry(arc.inner, arc.outer, 160, 1, 0, Math.PI), material);
    bow.position.set(0, -8, -62);
    context.world.add(bow);
  });
}

/** Slightly bent, tapered trunk with soft bark grooves; one unit tall, scaled per tree. */
function rainforestTrunkGeometry(radialSegments: number): THREE.BufferGeometry {
  const rings = 6;
  const positions: number[] = [];
  const indices: number[] = [];

  for (let ring = 0; ring <= rings; ring += 1) {
    const t = ring / rings;
    const taper = 1 - t * 0.56 + Math.sin(t * Math.PI) * 0.07;
    const bendX = Math.sin(t * Math.PI * 1.15) * 0.055 + t * t * 0.025;
    const bendZ = Math.sin(t * Math.PI * 0.8 + 0.7) * 0.035;
    for (let segment = 0; segment <= radialSegments; segment += 1) {
      const angle = (segment / radialSegments) * Math.PI * 2;
      const groove =
        1 + Math.sin(angle * 5 + t * 7) * 0.045 + Math.sin(angle * 9 - t * 4) * 0.018;
      positions.push(
        bendX + Math.cos(angle) * taper * groove,
        t,
        bendZ + Math.sin(angle) * taper * groove,
      );
    }
  }

  const stride = radialSegments + 1;
  for (let ring = 0; ring < rings; ring += 1) {
    for (let segment = 0; segment < radialSegments; segment += 1) {
      const a = ring * stride + segment;
      const b = a + stride;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Lance-shaped leaf with a soft curl and a dark-to-light gradient. */
function rainforestLeafGeometry(): THREE.BufferGeometry {
  const steps = 5;
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const root = new THREE.Color('#1f512f');
  const tip = new THREE.Color('#92c765');

  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    const halfWidth = Math.sin(Math.PI * t) * 0.17 * (1 - t * 0.2);
    const curl = Math.sin(Math.PI * t) * 0.05;
    const shade = root.clone().lerp(tip, t);
    positions.push(
      t,
      curl,
      -halfWidth,
      t,
      curl + 0.035 * Math.sin(Math.PI * t),
      0,
      t,
      curl,
      halfWidth,
    );
    for (let vertex = 0; vertex < 3; vertex += 1) colors.push(shade.r, shade.g, shade.b);
  }

  for (let step = 0; step < steps; step += 1) {
    const a = step * 3;
    const b = (step + 1) * 3;
    indices.push(a, a + 1, b, a + 1, b + 1, b);
    indices.push(a + 1, a + 2, b + 1, a + 2, b + 2, b + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

interface RainforestTree {
  x: number;
  z: number;
  radius: number;
}

interface RainforestVegetation {
  flowers: THREE.Vector3[];
  trees: RainforestTree[];
}

function addRainforestVegetation(context: WorldContext, night: boolean): RainforestVegetation {
  const random = seededRandom(night ? 7411 : 2289);
  const treeLimit = context.compact ? 36 : 94;
  const branchCount = context.compact ? 2 : 3;
  const rootCount = context.compact ? 2 : 3;
  const crownPerTree = context.compact ? 4 : 6;
  const leavesPerTree = context.compact ? 6 : 10;

  const trunks = new THREE.InstancedMesh(
    rainforestTrunkGeometry(context.compact ? 7 : 9),
    new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.98 }),
    treeLimit,
  );
  const branches = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.55, 1, 1, 6).translate(0, 0.5, 0),
    new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.98 }),
    treeLimit * (branchCount * 2 + rootCount),
  );
  const canopies = new THREE.InstancedMesh(
    organicGeometry(
      1,
      context.compact ? 1 : 2,
      night ? 17 : 29,
      night ? '#12291d' : '#1e4b2c',
      night ? '#3f7049' : '#82b55c',
      0.22,
    ),
    new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.94, vertexColors: true }),
    treeLimit * crownPerTree,
  );
  const leaves = new THREE.InstancedMesh(
    rainforestLeafGeometry(),
    new THREE.MeshStandardMaterial({
      color: '#ffffff',
      roughness: 0.9,
      side: THREE.DoubleSide,
      vertexColors: true,
    }),
    treeLimit * leavesPerTree,
  );

  const dummy = new THREE.Object3D();
  const up = new THREE.Vector3(0, 1, 0);
  const leafAxis = new THREE.Vector3(1, 0, 0);
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const color = new THREE.Color();

  let treeIndex = 0;
  let branchIndex = 0;
  let canopyIndex = 0;
  let leafIndex = 0;
  const trees: RainforestTree[] = [];

  const barkTint = () => {
    color.setHSL(
      0.065 + random() * 0.025,
      0.22 + random() * 0.1,
      night ? 0.15 + random() * 0.07 : 0.31 + random() * 0.1,
    );
    return color;
  };
  const foliageTint = () => {
    color.setHSL(
      0.26 + random() * 0.11,
      0.32 + random() * 0.2,
      night ? 0.68 + random() * 0.12 : 0.88 + random() * 0.12,
    );
    return color;
  };
  const addBranch = (from: THREE.Vector3, to: THREE.Vector3, radius: number) => {
    const direction = to.clone().sub(from);
    const length = direction.length();
    if (length < 0.02) return;
    dummy.position.copy(from);
    dummy.quaternion.setFromUnitVectors(up, direction.normalize());
    dummy.scale.set(radius, length, radius);
    dummy.updateMatrix();
    branches.setMatrixAt(branchIndex, dummy.matrix);
    branches.setColorAt(branchIndex, barkTint());
    branchIndex += 1;
  };

  for (let attempt = 0; attempt < treeLimit * 10 && treeIndex < treeLimit; attempt += 1) {
    const x = (random() * 2 - 1) * 36;
    const z = -50 + random() * 70;
    if (Math.abs(x - rainforestRiverX(z)) < 6.3) continue;
    if (Math.hypot(x, z - 10) < 10) continue;
    const y = rainforestHeight(x, z);
    if (y < -0.1) continue;

    const treeHeight = (8 + random() * 14) * (night ? 0.94 : 1);
    const trunkRadius = treeHeight * (0.026 + random() * 0.012);
    const base = new THREE.Vector3(x, y - 0.25, z);
    euler.set((random() - 0.5) * 0.075, random() * Math.PI * 2, (random() - 0.5) * 0.075);
    quaternion.setFromEuler(euler);

    trunks.setMatrixAt(
      treeIndex,
      matrix.compose(base, quaternion, scale.set(trunkRadius, treeHeight, trunkRadius)),
    );
    trunks.setColorAt(treeIndex, barkTint());
    trees.push({ x, z, radius: trunkRadius });
    treeIndex += 1;

    const trunkPoint = (t: number) =>
      new THREE.Vector3(0, treeHeight * t, 0).applyQuaternion(quaternion).add(base);

    for (let root = 0; root < rootCount; root += 1) {
      const angle = (root / rootCount) * Math.PI * 2 + random() * 0.7;
      const reach = trunkRadius * (3.2 + random() * 2.4);
      addBranch(
        trunkPoint(0.05),
        new THREE.Vector3(x + Math.cos(angle) * reach, y + 0.02, z + Math.sin(angle) * reach),
        trunkRadius * 0.42,
      );
    }

    const canopyCenters: THREE.Vector3[] = [];
    const canopyRadii: number[] = [];
    for (let cluster = 0; cluster < crownPerTree; cluster += 1) {
      const angle = random() * Math.PI * 2;
      const spread = treeHeight * (0.07 + random() * 0.13);
      const crownRadius = treeHeight * (0.12 + random() * 0.06);
      const center = trunkPoint(0.86).add(
        new THREE.Vector3(
          Math.cos(angle) * spread,
          (random() - 0.5) * treeHeight * 0.12,
          Math.sin(angle) * spread,
        ),
      );
      canopyCenters.push(center);
      canopyRadii.push(crownRadius);

      dummy.position.copy(center);
      dummy.quaternion.setFromEuler(
        new THREE.Euler((random() - 0.5) * 0.5, random() * Math.PI * 2, (random() - 0.5) * 0.5),
      );
      dummy.scale.set(
        crownRadius * (1.12 + random() * 0.32),
        crownRadius * (0.62 + random() * 0.18),
        crownRadius * (1.02 + random() * 0.28),
      );
      dummy.updateMatrix();
      canopies.setMatrixAt(canopyIndex, dummy.matrix);
      canopies.setColorAt(canopyIndex, foliageTint());
      canopyIndex += 1;
    }

    for (let branch = 0; branch < branchCount; branch += 1) {
      const target = canopyCenters[(branch * 2 + treeIndex) % canopyCenters.length];
      const start = trunkPoint(0.58 + random() * 0.11);
      const middle = start.clone().lerp(target, 0.52);
      middle.y += treeHeight * (0.025 + random() * 0.025);
      middle.x += (random() - 0.5) * treeHeight * 0.04;
      middle.z += (random() - 0.5) * treeHeight * 0.04;
      addBranch(start, middle, trunkRadius * 0.5);
      addBranch(middle, target, trunkRadius * 0.3);
    }

    for (let leaf = 0; leaf < leavesPerTree; leaf += 1) {
      const clusterIndex = leaf % canopyCenters.length;
      const center = canopyCenters[clusterIndex];
      const radius = canopyRadii[clusterIndex];
      const angle = random() * Math.PI * 2;
      const elevation = (random() - 0.35) * 0.95;
      const direction = new THREE.Vector3(
        Math.cos(angle) * Math.cos(elevation),
        Math.sin(elevation),
        Math.sin(angle) * Math.cos(elevation),
      ).normalize();
      dummy.position.copy(center).addScaledVector(direction, radius * (0.78 + random() * 0.18));
      dummy.quaternion.setFromUnitVectors(leafAxis, direction);
      dummy.rotateX((random() - 0.5) * 1.2);
      const length = radius * (0.42 + random() * 0.32);
      dummy.scale.set(length, length, length * (0.8 + random() * 0.4));
      dummy.updateMatrix();
      leaves.setMatrixAt(leafIndex, dummy.matrix);
      leaves.setColorAt(leafIndex, foliageTint());
      leafIndex += 1;
    }
  }

  (
    [
      [trunks, treeIndex],
      [branches, branchIndex],
      [canopies, canopyIndex],
      [leaves, leafIndex],
    ] as Array<[THREE.InstancedMesh, number]>
  ).forEach(([mesh, count]) => {
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.frustumCulled = false;
    if (!context.compact) mesh.castShadow = true;
    context.world.add(mesh);
  });

  const bushLimit = context.compact ? 64 : 150;
  const bushes = new THREE.InstancedMesh(
    organicGeometry(
      1,
      context.compact ? 0 : 1,
      night ? 41 : 53,
      night ? '#123321' : '#255d36',
      night ? '#3f7049' : '#7fae58',
      0.26,
    ),
    new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.94, vertexColors: true }),
    bushLimit,
  );
  const flowerGeometry = new THREE.SphereGeometry(0.14, 6, 5);
  const flowers = new THREE.InstancedMesh(
    flowerGeometry,
    new THREE.MeshStandardMaterial({
      color: '#ffffff',
      roughness: 0.5,
      emissive: new THREE.Color(night ? '#221100' : '#000000'),
      emissiveIntensity: night ? 0.8 : 0,
    }),
    bushLimit,
  );
  const palette = night
    ? ['#ffd23c', '#9ff3bf', '#ff7ab8', '#fff0a0']
    : ['#ff3b3b', '#ff8a1c', '#ffd23c', '#ffffff'];
  const flowerPositions: THREE.Vector3[] = [];
  let bushIndex = 0;
  let flowerIndex = 0;

  for (let attempt = 0; attempt < bushLimit * 8 && bushIndex < bushLimit; attempt += 1) {
    const x = (random() * 2 - 1) * 34;
    const z = -48 + random() * 66;
    if (Math.abs(x - rainforestRiverX(z)) < 4.2) continue;
    const y = rainforestHeight(x, z);
    if (y < -0.05) continue;

    const width = 0.8 + random() * 1.5;
    const height = 0.45 + random() * 0.7;
    bushes.setMatrixAt(
      bushIndex,
      matrix.compose(
        position.set(x, y + height * 0.35, z),
        quaternion.identity(),
        scale.set(width, height, width),
      ),
    );
    color.setHSL(
      0.26 + random() * 0.11,
      0.3 + random() * 0.16,
      night ? 0.68 + random() * 0.12 : 0.86 + random() * 0.14,
    );
    bushes.setColorAt(bushIndex, color);
    bushIndex += 1;

    if (random() < 0.5) {
      const bloom = new THREE.Vector3(
        x + (random() - 0.5) * 0.7,
        y + height * 1.05,
        z + (random() - 0.5) * 0.7,
      );
      flowers.setMatrixAt(
        flowerIndex,
        matrix.compose(bloom, quaternion.identity(), scale.set(1, 1, 1)),
      );
      flowers.setColorAt(flowerIndex, new THREE.Color(palette[flowerIndex % palette.length]));
      flowerPositions.push(bloom);
      flowerIndex += 1;
    }
  }

  (
    [
      [bushes, bushIndex],
      [flowers, flowerIndex],
    ] as Array<[THREE.InstancedMesh, number]>
  ).forEach(([mesh, count]) => {
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.frustumCulled = false;
    if (!context.compact) mesh.castShadow = true;
    context.world.add(mesh);
  });

  if (flowerPositions.length === 0) flowerPositions.push(new THREE.Vector3(0, 1, -8));
  return { flowers: flowerPositions, trees };
}

function addRainforestFireflies(context: WorldContext, night: boolean): void {
  const random = seededRandom(night ? 3301 : 1129);
  const count = context.compact ? 24 : 64;
  const positions = new Float32Array(count * 3);
  const origins = new Float32Array(count * 3);
  const phases = new Float32Array(count);

  for (let index = 0; index < count; index += 1) {
    const x = (random() * 2 - 1) * 28;
    const z = -46 + random() * 64;
    const y = Math.max(rainforestHeight(x, z), -0.6) + 0.7 + random() * 4.5;
    positions.set([x, y, z], index * 3);
    origins.set([x, y, z], index * 3);
    phases[index] = random() * Math.PI * 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: night ? '#ffe98a' : '#fff0b8',
    size: night ? 0.42 : 0.28,
    map: context.glow,
    transparent: true,
    opacity: night ? 0.92 : 0.48,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const fireflies = new THREE.Points(geometry, material);
  fireflies.frustumCulled = false;
  context.world.add(fireflies);
  context.update.push((time) => {
    for (let index = 0; index < count; index += 1) {
      const phase = phases[index];
      positions[index * 3] = origins[index * 3] + Math.sin(time * 0.55 + phase) * 1.1;
      positions[index * 3 + 1] = origins[index * 3 + 1] + Math.sin(time * 0.9 + phase * 1.7) * 0.6;
      positions[index * 3 + 2] = origins[index * 3 + 2] + Math.cos(time * 0.45 + phase) * 1.1;
    }
    geometry.attributes.position.needsUpdate = true;
  });
}

/** Smooth fore/hind wing outline in the butterfly's horizontal X-Z plane. */
function butterflyWingGeometry(forewing: boolean): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  if (forewing) {
    shape.moveTo(0.02, 0.02);
    shape.bezierCurveTo(0.28, 0.18, 0.66, 0.36, 0.96, 0.24);
    shape.bezierCurveTo(1.08, 0.11, 0.95, -0.09, 0.72, -0.21);
    shape.bezierCurveTo(0.45, -0.34, 0.16, -0.26, 0.02, -0.06);
  } else {
    shape.moveTo(0.02, -0.02);
    shape.bezierCurveTo(0.3, -0.02, 0.63, -0.11, 0.82, -0.29);
    shape.bezierCurveTo(0.99, -0.47, 0.76, -0.69, 0.5, -0.7);
    shape.bezierCurveTo(0.24, -0.69, 0.06, -0.42, 0.02, -0.12);
  }
  shape.closePath();

  const geometry = new THREE.ShapeGeometry(shape, 10);
  geometry.rotateX(Math.PI / 2);
  const positions = geometry.attributes.position as THREE.BufferAttribute;
  const colors: number[] = [];
  const root = new THREE.Color('#4b4b4b');
  const middle = new THREE.Color('#d2d2d2');
  const edge = new THREE.Color('#747474');

  for (let index = 0; index < positions.count; index += 1) {
    const x = positions.getX(index);
    const z = positions.getZ(index);
    positions.setY(index, Math.sin(Math.min(1, x) * Math.PI) * 0.03 + Math.sin(z * 2.2) * 0.012);
    const shade = root.clone().lerp(middle, THREE.MathUtils.clamp(x * 1.15, 0, 1));
    shade.lerp(edge, THREE.MathUtils.clamp((Math.abs(z) - 0.24) * 0.9, 0, 0.45));
    colors.push(shade.r, shade.g, shade.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  positions.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

function addRainforestButterflies(
  context: WorldContext,
  flowerPositions: THREE.Vector3[],
  night: boolean,
): void {
  const random = seededRandom(449);
  const count = context.compact ? 3 : 8;
  const foreGeometry = butterflyWingGeometry(true);
  const hindGeometry = butterflyWingGeometry(false);
  const spotGeometry = new THREE.CircleGeometry(0.055, 12).rotateX(-Math.PI / 2);
  const veinMaterial = new THREE.LineBasicMaterial({
    color: '#241a16',
    transparent: true,
    opacity: 0.46,
  });
  const palettes = [
    { wing: '#3d7fca', spot: '#d9ecff' },
    { wing: '#dd8a3a', spot: '#fff0c2' },
    { wing: '#d8bf4a', spot: '#fff6cf' },
  ].map(({ wing, spot }) => ({
    wing: new THREE.MeshStandardMaterial({
      color: wing,
      roughness: 0.64,
      side: THREE.DoubleSide,
      vertexColors: true,
    }),
    spot: new THREE.MeshBasicMaterial({
      color: spot,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.88,
      depthWrite: false,
    }),
  }));

  const buildWing = (forewing: boolean, palette: (typeof palettes)[number]) => {
    const group = new THREE.Group();
    group.add(new THREE.Mesh(forewing ? foreGeometry : hindGeometry, palette.wing));

    const spots = forewing
      ? ([
          [0.58, 0.1, 1],
          [0.78, -0.04, 0.75],
        ] as Array<[number, number, number]>)
      : ([
          [0.46, -0.36, 0.85],
          [0.62, -0.5, 0.7],
        ] as Array<[number, number, number]>);
    spots.forEach(([x, z, size]) => {
      const spot = new THREE.Mesh(spotGeometry, palette.spot);
      spot.position.set(x, 0.055, z);
      spot.scale.setScalar(size);
      group.add(spot);
    });

    const veins = forewing
      ? [
          [0.04, 0.01, 0.88, 0.2],
          [0.05, -0.02, 0.72, -0.17],
          [0.32, 0.11, 0.6, -0.24],
        ]
      : [
          [0.04, -0.04, 0.74, -0.34],
          [0.05, -0.06, 0.5, -0.62],
        ];
    veins.forEach(([x1, z1, x2, z2]) => {
      const vein = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x1, 0.042, z1),
          new THREE.Vector3(x2, 0.042, z2),
        ]),
        veinMaterial,
      );
      group.add(vein);
    });
    return group;
  };

  const prototypes = palettes.map((palette) => ({
    fore: buildWing(true, palette),
    hind: buildWing(false, palette),
  }));
  const bodyGeometry = new THREE.SphereGeometry(1, 10, 8);
  const bodyMaterial = standard('#1a1512', 0.72);
  const antennaMaterial = new THREE.LineBasicMaterial({
    color: '#2a211b',
    transparent: true,
    opacity: 0.82,
  });

  const butterflies: Array<{
    group: THREE.Group;
    home: THREE.Vector3;
    phase: number;
    radius: number;
    speed: number;
    beat: number;
    fore: Array<{ pivot: THREE.Group; side: number }>;
    hind: Array<{ pivot: THREE.Group; side: number }>;
  }> = [];

  for (let index = 0; index < count; index += 1) {
    const group = new THREE.Group();
    const abdomen = new THREE.Mesh(bodyGeometry, bodyMaterial);
    abdomen.scale.set(0.042, 0.042, 0.2);
    abdomen.position.z = -0.05;
    group.add(abdomen);
    const thorax = new THREE.Mesh(bodyGeometry, bodyMaterial);
    thorax.scale.set(0.055, 0.055, 0.09);
    thorax.position.z = 0.05;
    group.add(thorax);
    const head = new THREE.Mesh(bodyGeometry, bodyMaterial);
    head.scale.setScalar(0.045);
    head.position.z = 0.15;
    group.add(head);

    [-1, 1].forEach((side) => {
      group.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(side * 0.02, 0.02, 0.17),
            new THREE.Vector3(side * 0.05, 0.07, 0.24),
            new THREE.Vector3(side * 0.075, 0.1, 0.31),
          ]),
          antennaMaterial,
        ),
      );
    });

    const prototype = prototypes[index % prototypes.length];
    const fore: Array<{ pivot: THREE.Group; side: number }> = [];
    const hind: Array<{ pivot: THREE.Group; side: number }> = [];
    [-1, 1].forEach((side) => {
      const forePivot = new THREE.Group();
      forePivot.position.set(side * 0.04, 0.005, 0.02);
      forePivot.scale.x = side;
      forePivot.add(prototype.fore.clone());
      group.add(forePivot);
      fore.push({ pivot: forePivot, side });

      const hindPivot = new THREE.Group();
      hindPivot.position.set(side * 0.04, 0, -0.02);
      hindPivot.scale.x = side;
      hindPivot.add(prototype.hind.clone());
      group.add(hindPivot);
      hind.push({ pivot: hindPivot, side });
    });

    group.scale.setScalar(0.42 + random() * 0.16);
    context.world.add(group);

    if (night) {
      const rest = flowerPositions[index % flowerPositions.length].clone();
      group.position.set(
        rest.x + (random() - 0.5) * 0.08,
        rest.y + 0.16,
        rest.z + (random() - 0.5) * 0.08,
      );
      group.rotation.set(0.1, random() * Math.PI * 2, 0.04);
      fore.forEach(({ pivot, side }) => {
        pivot.rotation.set(0, side * 0.02, side * (Math.PI / 2 - 0.06));
      });
      hind.forEach(({ pivot, side }) => {
        pivot.rotation.set(0, -side * 0.08, side * (Math.PI / 2 - 0.12));
      });
      continue;
    }

    butterflies.push({
      group,
      home: flowerPositions[index % flowerPositions.length].clone(),
      phase: random() * Math.PI * 2,
      radius: 1.4 + random() * 2.0,
      speed: 0.22 + random() * 0.16,
      beat: 9 + index * 0.35,
      fore,
      hind,
    });
  }

  context.update.push((time) => {
    if (night) return;
    butterflies.forEach((butterfly) => {
      const positionAt = (t: number) => {
        const angle = t * butterfly.speed + butterfly.phase;
        return new THREE.Vector3(
          butterfly.home.x +
            Math.sin(angle) * butterfly.radius +
            Math.sin(angle * 2.3 + butterfly.phase) * 0.45,
          butterfly.home.y +
            1.15 +
            Math.sin(angle * 1.7 + butterfly.phase) * 0.55 +
            Math.sin(angle * 3.1) * 0.1,
          butterfly.home.z +
            Math.cos(angle * 0.83) * butterfly.radius * 0.85 +
            Math.sin(angle * 1.3) * 0.32,
        );
      };
      const current = positionAt(time);
      const travel = positionAt(time + 0.12).sub(current);
      butterfly.group.position.copy(current);
      butterfly.group.rotation.y = Math.atan2(travel.x, travel.z);
      butterfly.group.rotation.x =
        -Math.atan2(travel.y, Math.hypot(travel.x, travel.z)) * 0.45;
      butterfly.group.rotation.z =
        Math.sin(time * butterfly.speed * 2 + butterfly.phase) * 0.16;

      const beatTime = time * butterfly.beat + butterfly.phase;
      const foreStroke = Math.sin(beatTime);
      const foreLift = 0.14 + 0.94 * Math.pow((foreStroke + 1) / 2, 0.72);
      const hindStroke = Math.sin(beatTime - 0.42);
      const hindLift = 0.12 + 0.86 * Math.pow((hindStroke + 1) / 2, 0.72);
      butterfly.fore.forEach(({ pivot, side }) => {
        pivot.rotation.z = side * foreLift;
        pivot.rotation.y = side * (0.055 + Math.sin(beatTime) * 0.05);
      });
      butterfly.hind.forEach(({ pivot, side }) => {
        pivot.rotation.z = side * hindLift;
        pivot.rotation.y = side * Math.sin(beatTime - 0.7) * 0.045;
      });
    });
  });
}

function addRainforestMacaws(context: WorldContext): void {
  const random = seededRandom(663);
  const red = standard('#d51818', 0.6);
  const blue = standard('#1a5fd0', 0.6);
  const yellow = standard('#f5c518', 0.6);
  const pale = standard('#f6f1e6', 0.7);
  const dark = standard('#151515', 0.5);
  const sphereGeometry = new THREE.SphereGeometry(1, 10, 8);
  const birds: Array<{
    group: THREE.Group;
    centerX: number;
    centerZ: number;
    radius: number;
    speed: number;
    phase: number;
    y: number;
    wings: THREE.Group[];
  }> = [];

  const count = context.compact ? 1 : 3;
  for (let index = 0; index < count; index += 1) {
    const group = new THREE.Group();
    const ellipsoid = (
      material: THREE.Material,
      scales: [number, number, number],
      offsets: [number, number, number],
    ) => {
      const mesh = new THREE.Mesh(sphereGeometry, material);
      mesh.scale.set(...scales);
      mesh.position.set(...offsets);
      group.add(mesh);
      return mesh;
    };
    ellipsoid(red, [0.2, 0.2, 0.45], [0, 0, 0]);
    ellipsoid(red, [0.17, 0.17, 0.18], [0, 0.08, 0.46]);
    ellipsoid(pale, [0.1, 0.09, 0.08], [0, 0.03, 0.57]);
    [-1, 1].forEach((side) => {
      ellipsoid(dark, [0.03, 0.03, 0.03], [0.1 * side, 0.12, 0.55]);
    });
    const beak = new THREE.Mesh(
      new THREE.ConeGeometry(0.07, 0.24, 8).rotateX(Math.PI / 2).translate(0, -0.03, 0.68),
      pale,
    );
    group.add(beak);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 1.1), red);
    tail.position.z = -0.95;
    group.add(tail);
    const tailTip = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.03, 0.55), blue);
    tailTip.position.z = -1.6;
    group.add(tailTip);

    const wings = [-1, 1].map((side) => {
      const pivot = new THREE.Group();
      pivot.position.set(0.16 * side, 0.08, 0.05);
      const inner = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.04, 0.6), red);
      inner.position.x = 0.28 * side;
      const middle = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.04, 0.55), yellow);
      middle.position.set(0.75 * side, 0, -0.05);
      const outer = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.04, 0.5), blue);
      outer.position.set(1.25 * side, 0, -0.12);
      pivot.add(inner, middle, outer);
      group.add(pivot);
      return pivot;
    });

    group.scale.setScalar(context.compact ? 1.2 : 1.7);
    context.world.add(group);
    birds.push({
      group,
      centerX: [-5, 4, 0][index],
      centerZ: [-20, -26, -13][index],
      radius: 3.5 + random() * 2,
      speed: index % 2 === 0 ? 0.34 : -0.28,
      phase: index * 2.1,
      y: 8 + random() * 4,
      wings,
    });
  }

  context.update.push((time) => {
    birds.forEach((bird) => {
      const angle = time * bird.speed + bird.phase;
      const direction = Math.sign(bird.speed) || 1;
      bird.group.position.set(
        bird.centerX + bird.radius * Math.cos(angle),
        bird.y + Math.sin(angle * 2.3) * 1.2,
        bird.centerZ + bird.radius * Math.sin(angle),
      );
      bird.group.rotation.set(
        0,
        Math.atan2(-Math.sin(angle) * direction, Math.cos(angle) * direction),
        0.28 * direction,
      );
      const flap =
        Math.sin(time * 8 + bird.phase) *
        (0.22 + 0.45 * Math.max(0, Math.sin(time * 0.5 + bird.phase * 2)));
      bird.wings[0].rotation.z = -flap;
      bird.wings[1].rotation.z = flap;
    });
  });
}

function addRainforestFrogs(context: WorldContext, night: boolean): void {
  const random = seededRandom(557);
  const padGeometry = new THREE.CylinderGeometry(0.72, 0.72, 0.05, 14);
  const padMaterial = standard(night ? '#2f7a46' : '#4f9b45', 0.66);
  const bloomGeometry = new THREE.SphereGeometry(0.14, 6, 5);
  const bloomMaterial = standard('#ff7ab8', 0.5);
  const pads: THREE.Vector3[] = [];
  const padCount = context.compact ? 6 : 11;

  for (let index = 0; index < padCount; index += 1) {
    const z = -40 + (index * 58) / (padCount - 1);
    const x = rainforestRiverX(z) + (random() * 2 - 1) * 2.8;
    const pad = new THREE.Mesh(padGeometry, padMaterial);
    pad.position.set(x, -0.74, z);
    pad.rotation.y = random() * Math.PI * 2;
    pad.scale.setScalar(0.75 + random() * 0.55);
    context.world.add(pad);
    pads.push(pad.position.clone());
    if (index % 3 === 0) {
      const bloom = new THREE.Mesh(bloomGeometry, bloomMaterial);
      bloom.position.set(x + 0.2, -0.6, z + 0.1);
      context.world.add(bloom);
    }
  }

  const frogGeometry = new THREE.SphereGeometry(1, 10, 8);
  const green = standard(night ? '#3fae35' : '#58c42a', 0.45);
  const cream = standard('#eef3c8', 0.7);
  const red = standard('#e21219', 0.3);
  const dark = standard('#111111', 0.6);
  const orange = standard('#ff7a1a', 0.6);
  const frogs: Array<{
    group: THREE.Group;
    padIndex: number;
    from: THREE.Vector3;
    to: THREE.Vector3;
    progress: number;
    wait: number;
  }> = [];
  const frogCount = context.compact ? 1 : 3;

  for (let index = 0; index < frogCount; index += 1) {
    const group = new THREE.Group();
    const ellipsoid = (
      material: THREE.Material,
      scales: [number, number, number],
      offsets: [number, number, number],
    ) => {
      const mesh = new THREE.Mesh(frogGeometry, material);
      mesh.scale.set(...scales);
      mesh.position.set(...offsets);
      group.add(mesh);
    };
    ellipsoid(green, [0.28, 0.2, 0.38], [0, 0.2, 0]);
    ellipsoid(cream, [0.2, 0.1, 0.28], [0, 0.1, 0.05]);
    [-1, 1].forEach((side) => {
      ellipsoid(red, [0.1, 0.1, 0.1], [0.14 * side, 0.4, 0.22]);
      ellipsoid(dark, [0.03, 0.07, 0.03], [0.14 * side, 0.4, 0.31]);
      ellipsoid(orange, [0.1, 0.04, 0.15], [0.2 * side, 0.04, 0.3]);
      ellipsoid(orange, [0.12, 0.04, 0.2], [0.22 * side, 0.04, -0.2]);
    });
    group.scale.setScalar(1.15);
    const padIndex = (index * 2) % pads.length;
    group.position.copy(pads[padIndex]);
    group.position.y += 0.06;
    context.world.add(group);
    frogs.push({
      group,
      padIndex,
      from: new THREE.Vector3(),
      to: new THREE.Vector3(),
      progress: 1,
      wait: 1 + index * 0.9,
    });
  }

  context.update.push((_time, delta) => {
    frogs.forEach((frog, index) => {
      if (frog.progress >= 1) {
        frog.wait -= delta;
        if (frog.wait <= 0) {
          const nextPad = (frog.padIndex + 2 + index) % pads.length;
          frog.from.copy(frog.group.position);
          frog.to.copy(pads[nextPad]);
          frog.to.y += 0.06;
          frog.padIndex = nextPad;
          frog.progress = 0;
          frog.group.rotation.y = Math.atan2(
            frog.to.x - frog.from.x,
            frog.to.z - frog.from.z,
          );
        }
        return;
      }
      frog.progress = Math.min(1, frog.progress + delta / 0.85);
      frog.group.position.lerpVectors(frog.from, frog.to, frog.progress);
      frog.group.position.y += Math.sin(frog.progress * Math.PI) * 0.85;
      frog.group.rotation.x = -Math.sin(frog.progress * Math.PI) * 0.35;
      if (frog.progress >= 1) {
        frog.wait = 2 + ((index * 1.7) % 2.5);
        frog.group.rotation.x = 0;
      }
    });
  });
}

/** Fine, deterministic fur strokes; grayscale so the coat tint still reads. */
function bearFurTexture(context: WorldContext): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 96;
  canvas.height = 96;
  const paint = canvas.getContext('2d');
  if (paint) {
    paint.fillStyle = '#ffffff';
    paint.fillRect(0, 0, 96, 96);
    const random = seededRandom(311);
    for (let index = 0; index < 460; index += 1) {
      const x = random() * 96;
      const y = random() * 96;
      const length = 3 + random() * 7;
      const light = random() > 0.44;
      paint.strokeStyle = light
        ? `rgba(255,255,255,${0.07 + random() * 0.1})`
        : `rgba(0,0,0,${0.035 + random() * 0.075})`;
      paint.lineWidth = 0.5 + random() * 0.8;
      paint.beginPath();
      paint.moveTo(x, y);
      paint.lineTo(x + (random() - 0.5) * 3.2, y + length);
      paint.stroke();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2.4, 2.4);
  texture.colorSpace = THREE.SRGBColorSpace;
  context.textures.push(texture);
  return texture;
}

/** One continuous lofted torso instead of overlapping spheres, with a shoulder hump. */
function bearBodyGeometry(): THREE.BufferGeometry {
  const radialSegments = 14;
  const rings = 16;
  const controls: Array<[number, number, number, number]> = [
    [0, 0.44, 0.42, 1.14],
    [0.14, 0.63, 0.58, 1.16],
    [0.38, 0.72, 0.62, 1.16],
    [0.58, 0.79, 0.69, 1.31],
    [0.74, 0.68, 0.58, 1.29],
    [0.9, 0.5, 0.45, 1.33],
    [1, 0.36, 0.34, 1.36],
  ];
  const positions: number[] = [];
  const colors: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const profile = (t: number) => {
    let index = 0;
    while (index < controls.length - 2 && t > controls[index + 1][0]) index += 1;
    const from = controls[index];
    const to = controls[index + 1];
    const raw = THREE.MathUtils.clamp((t - from[0]) / (to[0] - from[0]), 0, 1);
    const blend = raw * raw * (3 - 2 * raw);
    return {
      radiusX: THREE.MathUtils.lerp(from[1], to[1], blend),
      radiusY: THREE.MathUtils.lerp(from[2], to[2], blend),
      centerY: THREE.MathUtils.lerp(from[3], to[3], blend),
    };
  };

  for (let ring = 0; ring <= rings; ring += 1) {
    const t = ring / rings;
    const { radiusX, radiusY, centerY } = profile(t);
    const z = -1.18 + t * 2.28;
    for (let segment = 0; segment <= radialSegments; segment += 1) {
      const angle = (segment / radialSegments) * Math.PI * 2;
      const fur =
        1 + Math.sin(angle * 6 + t * 11) * 0.018 + Math.sin(angle * 11 - t * 7) * 0.01;
      positions.push(
        Math.cos(angle) * radiusX * fur,
        centerY + Math.sin(angle) * radiusY * fur,
        z,
      );
      const shade =
        0.78 + Math.max(0, Math.sin(angle)) * 0.17 + Math.sin(t * Math.PI) * 0.04;
      colors.push(shade, shade * 0.98, shade * 0.94);
      uvs.push((segment / radialSegments) * 2.4, t * 2.4);
    }
  }

  const rearCenter = positions.length / 3;
  positions.push(0, profile(0).centerY, -1.18);
  colors.push(0.82, 0.8, 0.76);
  uvs.push(1.2, 0);
  const frontCenter = positions.length / 3;
  positions.push(0, profile(1).centerY, 1.1);
  colors.push(0.82, 0.8, 0.76);
  uvs.push(1.2, 2.4);

  const stride = radialSegments + 1;
  for (let ring = 0; ring < rings; ring += 1) {
    for (let segment = 0; segment < radialSegments; segment += 1) {
      const a = ring * stride + segment;
      const b = a + stride;
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  for (let segment = 0; segment < radialSegments; segment += 1) {
    indices.push(rearCenter, segment + 1, segment);
    const frontRing = rings * stride;
    indices.push(frontCenter, frontRing + segment, frontRing + segment + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Slightly bowed tapered leg segment, top at the pivot and bottom one unit below. */
function bearLimbGeometry(topRadius: number, bottomRadius: number): THREE.BufferGeometry {
  const radialSegments = 8;
  const rings = 6;
  const positions: number[] = [];
  const colors: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  for (let ring = 0; ring <= rings; ring += 1) {
    const t = ring / rings;
    const radius =
      THREE.MathUtils.lerp(topRadius, bottomRadius, t) * (1 + Math.sin(t * Math.PI) * 0.06);
    const bendX = Math.sin(t * Math.PI) * 0.025;
    const bendZ = Math.sin(t * Math.PI * 0.8) * 0.018;
    const shade = 0.82 + (1 - t) * 0.12;
    for (let segment = 0; segment <= radialSegments; segment += 1) {
      const angle = (segment / radialSegments) * Math.PI * 2;
      positions.push(
        bendX + Math.cos(angle) * radius,
        -t,
        bendZ + Math.sin(angle) * radius,
      );
      colors.push(shade, shade * 0.98, shade * 0.94);
      uvs.push((segment / radialSegments) * 1.6, t * 1.6);
    }
  }

  const stride = radialSegments + 1;
  for (let ring = 0; ring < rings; ring += 1) {
    for (let segment = 0; segment < radialSegments; segment += 1) {
      const a = ring * stride + segment;
      const b = a + stride;
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Precompute a lateral route along the right bank. Trees are avoided using their
 * real trunk positions plus root/bear clearance, so the bear arcs around them
 * instead of walking through them.
 */
function buildRainforestBearPath(trees: RainforestTree[]): number[] {
  const startZ = -32;
  const endZ = -4;
  const steps = 84;
  const path: number[] = [];
  let previousX = rainforestRiverX(startZ) + 7.6;

  for (let step = 0; step <= steps; step += 1) {
    const z = startZ + (step / steps) * (endZ - startZ);
    const riverX = rainforestRiverX(z);
    const baseX = riverX + 7.6;
    const minX = riverX + 5.6;
    const candidates = [baseX, previousX];

    trees.forEach((tree) => {
      const clearance = tree.radius * 3.4 + 1.25;
      if (Math.abs(z - tree.z) < clearance + 1.2) {
        candidates.push(tree.x + clearance + 0.12);
      }
    });

    let bestX = baseX;
    let bestCost = Number.POSITIVE_INFINITY;
    candidates.forEach((candidate) => {
      const x = Math.max(candidate, minX);
      let cost = Math.abs(x - baseX) * 0.55 + Math.abs(x - previousX) * 0.32;
      trees.forEach((tree) => {
        const clearance = tree.radius * 3.4 + 1.25;
        const distance = Math.hypot(x - tree.x, z - tree.z);
        if (distance < clearance) cost += (clearance - distance) * 14;
      });
      if (cost < bestCost) {
        bestCost = cost;
        bestX = x;
      }
    });

    path.push(bestX);
    previousX = bestX;
  }

  return path;
}

function pathXAt(path: number[], z: number): number {
  const startZ = -32;
  const endZ = -4;
  const position = THREE.MathUtils.clamp(
    ((z - startZ) / (endZ - startZ)) * (path.length - 1),
    0,
    path.length - 1,
  );
  const index = Math.min(Math.floor(position), path.length - 2);
  return THREE.MathUtils.lerp(path[index], path[index + 1], position - index);
}

function addRainforestBear(context: WorldContext, night: boolean, trees: RainforestTree[]): void {
  const bearPath = buildRainforestBearPath(trees);
  const fur = bearFurTexture(context);
  const coatColor = night ? '#2b231d' : '#3a2d25';
  const shadeColor = night ? '#201914' : '#281e18';
  const muzzleColor = night ? '#6f5948' : '#8b7159';
  const ringColor = night ? '#8f8170' : '#b9a688';
  const coatMaterial = new THREE.MeshStandardMaterial({
    color: coatColor,
    map: fur,
    vertexColors: true,
    roughness: 0.92,
  });
  const shadeMaterial = new THREE.MeshStandardMaterial({
    color: shadeColor,
    map: fur,
    vertexColors: true,
    roughness: 0.96,
  });
  const muzzleMaterial = new THREE.MeshStandardMaterial({
    color: muzzleColor,
    map: fur,
    roughness: 0.9,
  });
  const ringMaterial = standard(ringColor, 0.9);
  const dark = standard('#120d0a', 0.7);
  const glint = standard('#efe7d6', 0.4);

  const group = new THREE.Group();
  const body = new THREE.Group();
  group.add(body);
  body.add(new THREE.Mesh(bearBodyGeometry(), coatMaterial));

  const organicPart = (
    parent: THREE.Object3D,
    material: THREE.Material,
    seed: number,
    scales: [number, number, number],
    offsets: [number, number, number],
    detail = 1,
  ) => {
    const mesh = new THREE.Mesh(
      organicGeometry(1, detail, seed, '#ffffff', '#d8d8d8', 0.1),
      material,
    );
    mesh.scale.set(...scales);
    mesh.position.set(...offsets);
    parent.add(mesh);
    return mesh;
  };

  const head = new THREE.Group();
  head.position.set(0, 1.42, 1.34);
  body.add(head);
  organicPart(head, coatMaterial, 41, [0.4, 0.36, 0.42], [0, 0, 0], 2);
  organicPart(head, muzzleMaterial, 53, [0.29, 0.18, 0.24], [0, -0.08, 0.33]);
  organicPart(head, dark, 67, [0.11, 0.075, 0.06], [0, -0.03, 0.55], 0);

  [-1, 1].forEach((side) => {
    organicPart(head, coatMaterial, 71 + side, [0.14, 0.15, 0.09], [0.3 * side, 0.31, -0.05]);
    organicPart(head, shadeMaterial, 83 + side, [0.08, 0.09, 0.04], [0.3 * side, 0.31, 0.02]);
    const eyeRing = new THREE.Mesh(new THREE.TorusGeometry(0.105, 0.022, 6, 16), ringMaterial);
    eyeRing.position.set(0.18 * side, 0.08, 0.4);
    eyeRing.rotation.y = side * 0.12;
    head.add(eyeRing);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 6), dark);
    eye.position.set(0.18 * side, 0.08, 0.425);
    head.add(eye);
    const shine = new THREE.Mesh(new THREE.SphereGeometry(0.012, 6, 4), glint);
    shine.position.set(0.17 * side, 0.095, 0.451);
    head.add(shine);
  });

  const mouth = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -0.12, 0.58),
      new THREE.Vector3(0, -0.15, 0.64),
    ]),
    new THREE.LineBasicMaterial({ color: '#120d0a' }),
  );
  head.add(mouth);

  const upperGeometry = bearLimbGeometry(0.23, 0.18);
  const lowerGeometry = bearLimbGeometry(0.17, 0.15);
  const pawGeometry = organicGeometry(1, 1, 101, '#ffffff', '#d8d8d8', 0.12);
  const legs: Array<{ pivot: THREE.Group; lower: THREE.Group; offset: number }> = [];
  [
    [0.44, 0.62, 0],
    [-0.44, 0.62, Math.PI],
    [0.44, -0.65, Math.PI],
    [-0.44, -0.65, 0],
  ].forEach(([x, z, offset]) => {
    const pivot = new THREE.Group();
    pivot.position.set(x, 1.05, z);
    body.add(pivot);
    const upper = new THREE.Mesh(upperGeometry, coatMaterial);
    upper.scale.y = 0.72;
    pivot.add(upper);

    const lower = new THREE.Group();
    lower.position.y = -0.7;
    pivot.add(lower);
    const shin = new THREE.Mesh(lowerGeometry, shadeMaterial);
    shin.scale.y = 0.46;
    lower.add(shin);

    const paw = new THREE.Mesh(pawGeometry, coatMaterial);
    paw.scale.set(0.22, 0.12, 0.3);
    paw.position.set(0, -0.48, 0.09);
    lower.add(paw);
    [-0.12, 0, 0.12].forEach((clawX) => {
      lower.add(
        new THREE.Mesh(
          new THREE.ConeGeometry(0.035, 0.2, 5)
            .rotateX(Math.PI / 2)
            .translate(clawX, -0.52, 0.36),
          ringMaterial,
        ),
      );
    });
    legs.push({ pivot, lower, offset });
  });

  organicPart(body, shadeMaterial, 131, [0.1, 0.1, 0.14], [0, 1.18, -1.18]);

  group.scale.setScalar(night ? 0.95 : 1.05);
  context.world.add(group);
  context.update.push((time) => {
    const pace = night ? 0.055 : 0.09;
    const z = -18 + Math.sin(time * pace) * 12;
    const x = pathXAt(bearPath, z);
    group.position.set(x, Math.max(rainforestHeight(x, z), -0.5) + 0.04, z);
    const nextZ = -18 + Math.sin((time + 0.2) * pace) * 12;
    const nextX = pathXAt(bearPath, nextZ);
    group.rotation.y = Math.atan2(nextX - x, nextZ - z);

    const phase = time * (night ? 1.2 : 2.0);
    legs.forEach((leg) => {
      leg.pivot.rotation.x = Math.sin(phase + leg.offset) * 0.26;
      leg.lower.rotation.x = Math.max(0, -Math.cos(phase + leg.offset)) * 0.3;
    });
    body.position.y = Math.sin(phase * 2) * 0.018;
    body.rotation.z = Math.sin(phase) * 0.01;
    head.rotation.y = Math.sin(time * 0.34) * 0.12;
    head.rotation.x = Math.sin(time * 0.27) * 0.05 + (night ? 0.04 : 0);
  });
}

function buildRainforest(context: WorldContext, night: boolean, camera: THREE.PerspectiveCamera): void {
  addSky(
    context,
    night
      ? [[0, '#06100b'], [0.58, '#0b1a12'], [1, '#24432d']]
      : [[0, '#9fc7b0'], [0.58, '#d6e8d2'], [1, '#fff2c8']],
  );
  addRainforestTerrain(context, night);
  addRainforestRiver(context, night);
  addLights(context, {
    sky: night ? '#a9d8b8' : '#e8f7e2',
    ground: night ? '#0a1c12' : '#496b3d',
    sun: night ? '#cfe8ff' : '#fff0b8',
    sunPower: night ? 0.48 : 2.05,
    ambient: night ? 0.26 : 0.68,
  });
  addGlow(context, night ? '#bfe8ff' : '#fff0b8', [-12, 14, -36], night ? 18 : 30, night ? 0.32 : 0.36);
  if (night) addMoon(context, [11, 15, -38], 1.15);

  addRainforestLightShafts(context, night);
  addRainforestMist(context, night);
  const { flowers, trees } = addRainforestVegetation(context, night);
  addRainforestBow(context, night);
  addRain(context, context.compact ? 200 : 560, night ? '#d6e8ff' : '#eaf6ff');
  addRainforestFireflies(context, night);
  addRainforestButterflies(context, flowers, night);
  if (!night) addRainforestMacaws(context);
  addRainforestFrogs(context, night);
  if (!context.compact) addRainforestBear(context, night, trees);

  const target = new THREE.Vector3(0, 2.8, -14);
  const origin = camera.position.clone();
  context.update.push((time) => {
    const sway = Math.sin(time * 0.12) * (context.compact ? 0.5 : 1.15);
    camera.position.set(origin.x + sway, origin.y + Math.sin(time * 0.17) * 0.18, origin.z);
    camera.lookAt(target);
  });
}

export function createSceneWorld(key: SceneKey, compact: boolean): SceneWorld {
  const scene = new THREE.Scene();
  const winter = key.startsWith('winter');
  const rainy = key.startsWith('rainy');
  const fogColor = winter
    ? key.endsWith('night')
      ? '#142844'
      : '#c9dfef'
    : rainy
      ? key.endsWith('night')
        ? '#0b1a12'
        : '#a9c9b1'
      : '#9fb6cf';
  scene.fog = new THREE.Fog(
    new THREE.Color(fogColor),
    winter ? 32 : rainy ? 24 : 28,
    winter ? 150 : rainy ? 96 : 90,
  );
  const polarNight = key === 'winter-night';
  const antarcticDay = key === 'winter-day';
  const desertNight = key === 'summer-night';
  const camera = new THREE.PerspectiveCamera(
    antarcticDay ? 42 : polarNight ? 50 : desertNight ? 55 : rainy ? 44 : 38,
    1,
    desertNight ? 0.5 : 0.1,
    desertNight ? 3000 : 240,
  );
  if (polarNight) {
    camera.position.set(0, 6.5, 16);
    camera.lookAt(0, 5.5, -28);
  } else if (rainy) {
    camera.position.set(0, 5.4, 16);
    camera.lookAt(0, 2.8, -14);
  } else if (key === 'summer-day') {
    camera.position.set(0, 3.4, 10.5);
    camera.lookAt(0, 1.5, -3);
  }

  const context: WorldContext = {
    world: scene,
    textures: [],
    compact,
    glow: glowTexture(),
    update: [],
  };
  context.textures.push(context.glow);

  if (antarcticDay) buildWinter(context, false, camera);
  if (polarNight) buildWinter(context, true, camera);
  if (key === 'summer-day') buildSummer(context, false);
  if (desertNight) buildDesertNight(context, camera);
  if (key === 'rainy-day') buildRainforest(context, false, camera);
  if (key === 'rainy-night') buildRainforest(context, true, camera);

  if (polarNight) {
    const target = new THREE.Vector3(0, 5.5, -28);
    context.update.push((time) => {
      const turn = (Math.PI * 2 * time) / 12;
      camera.position.x = Math.sin(turn) * 0.8;
      camera.position.y = 6.5 + Math.sin(turn * 2) * 0.18;
      camera.lookAt(target);
    });
  }

  return {
    scene,
    camera,
    update: (time, delta) => {
      context.update.forEach((update) => update(time, delta));
    },
    resize: (width, height) => {
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    },
    dispose: () => {
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.Points || child instanceof THREE.Line || child instanceof THREE.Sprite) {
          child.geometry?.dispose();
          const materials = Array.isArray(child.material) ? child.material : [child.material];
          materials.forEach((material) => material?.dispose());
        }
      });
      context.textures.forEach((texture) => texture.dispose());
      scene.clear();
    },
  };
}
