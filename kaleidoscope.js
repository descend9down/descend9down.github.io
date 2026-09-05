/**
 * Shared WebGL kaleidoscope effect for the DESCEND9DOWN interactive pieces.
 *
 * Replaces the old per-piece Canvas 2D implementation, which allocated a
 * fresh offscreen canvas and re-drew the whole image into it every frame,
 * then stamped up to 18 rotated whole copies of that image via clipped
 * drawImage() calls per frame. This version uploads the source image as a
 * GPU texture once and does a true angle-fold mirror sample per pixel in a
 * single fragment shader pass — one draw call per frame regardless of
 * slice count, and a mathematically seamless mirror at each wedge boundary
 * instead of independently rotated copies.
 */
class KaleidoscopeFX {
  /**
   * @param {HTMLCanvasElement} canvas - target canvas, sized to its container
   * @param {HTMLImageElement} img - source image, already loaded
   * @param {Object} [opts]
   * @param {[number,number,number]} [opts.glowColor] - center glow RGB, 0-255 each
   */
  constructor(canvas, img, opts = {}) {
    this.canvas = canvas;
    this.img = img;
    this.glowColor = (opts.glowColor || [160, 60, 255]).map(c => c / 255);

    this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    this.slices = 8;
    this.active = false;
    this.time = 0;
    this.lastFrame = 0;
    this.raf = null;
    this.ready = false;

    this._u = {};
    if (this.gl) this._initGL();
  }

  _compileShader(type, src) {
    const gl = this.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('KaleidoscopeFX shader error:', gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  _initGL() {
    const gl = this.gl;
    const vsSrc = 'attribute vec2 a_pos; void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }';
    const fsSrc = `
      precision highp float;
      uniform vec2 u_resolution;
      uniform sampler2D u_tex;
      uniform vec2 u_imgSize;
      uniform vec2 u_imgOrigin;
      uniform float u_slices;
      uniform float u_time;
      uniform vec3 u_glowColor;
      #define PI 3.14159265359
      void main(){
        vec2 fragPx = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
        vec2 center = u_resolution * 0.5;
        vec2 p = fragPx - center;
        float r = length(p);
        float a = atan(p.y, p.x) + u_time;
        float slice = 2.0 * PI / u_slices;
        a = mod(a, slice);
        a = abs(a - slice * 0.5);
        vec2 samplePx = center + vec2(cos(a), sin(a)) * r;
        vec2 uv = (samplePx - u_imgOrigin) / u_imgSize;

        vec3 col = vec3(0.0);
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0){
          col = texture2D(u_tex, uv).rgb;
        }

        float radius = min(u_resolution.x, u_resolution.y) * 0.52;
        float vign = smoothstep(radius * 0.6, radius * 1.1, r);
        col = mix(col, vec3(0.0), vign);

        float glowT = smoothstep(60.0 * (u_resolution.x / 480.0), 0.0, r) * 0.25;
        col = mix(col, u_glowColor, glowT);

        gl_FragColor = vec4(col, 1.0);
      }
    `;

    const vs = this._compileShader(gl.VERTEX_SHADER, vsSrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSrc);
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('KaleidoscopeFX link error:', gl.getProgramInfoLog(program));
      return;
    }
    this.program = program;
    gl.useProgram(program);

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'a_pos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    this._u.resolution = gl.getUniformLocation(program, 'u_resolution');
    this._u.tex = gl.getUniformLocation(program, 'u_tex');
    this._u.imgSize = gl.getUniformLocation(program, 'u_imgSize');
    this._u.imgOrigin = gl.getUniformLocation(program, 'u_imgOrigin');
    this._u.slices = gl.getUniformLocation(program, 'u_slices');
    this._u.time = gl.getUniformLocation(program, 'u_time');
    this._u.glowColor = gl.getUniformLocation(program, 'u_glowColor');

    // Texture uploaded once — the source image never changes, unlike the
    // old implementation's per-frame offscreen canvas redraw.
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.ready = true;
    this.resize();
  }

  /** Call after the canvas element's on-screen size changes. */
  resize() {
    if (!this.gl) return;
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(this.canvas.clientWidth * dpr) || this.canvas.width;
    const h = Math.round(this.canvas.clientHeight * dpr) || this.canvas.height;
    this.canvas.width = w;
    this.canvas.height = h;
    gl.viewport(0, 0, w, h);

    const iW = this.img.naturalWidth, iH = this.img.naturalHeight;
    const scale = Math.max(w / iW, h / iH) * 0.8;
    this.imgSize = [iW * scale, iH * scale];
    this.imgOrigin = [w / 2 - this.imgSize[0] / 2, h / 2 - this.imgSize[1] / 2];
  }

  setSlices(n) {
    this.slices = n;
  }

  /** @returns {boolean} the new active state */
  toggle() {
    this.active ? this.stop() : this.start();
    return this.active;
  }

  start() {
    if (this.active || !this.ready) return;
    this.active = true;
    this.lastFrame = performance.now();
    this.raf = requestAnimationFrame((t) => this._loop(t));
  }

  stop() {
    this.active = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  _loop(now) {
    if (!this.active) return;
    const dt = (now - this.lastFrame) / 1000;
    this.lastFrame = now;
    this.time += dt * 0.24; // matches the original's perceived rotation speed
    this._draw();
    this.raf = requestAnimationFrame((t) => this._loop(t));
  }

  _draw() {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform2f(this._u.resolution, this.canvas.width, this.canvas.height);
    gl.uniform2f(this._u.imgSize, this.imgSize[0], this.imgSize[1]);
    gl.uniform2f(this._u.imgOrigin, this.imgOrigin[0], this.imgOrigin[1]);
    gl.uniform1f(this._u.slices, this.slices);
    gl.uniform1f(this._u.time, this.time);
    gl.uniform3f(this._u.glowColor, this.glowColor[0], this.glowColor[1], this.glowColor[2]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this._u.tex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
