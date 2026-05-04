/* ====================================================================
   diagram.js — tensor-flow diagram renderer

   Public API:
     renderDiagram(svg, spec)
       svg:  SVGElement or selector string (e.g. '#d')
       spec: {
         width:  number,                 // canvas width  (sets svg width + viewBox)
         height: number,                 // canvas height (sets svg height + viewBox)

         modules: [
           {id, label, x, y, w, h}      // dashed rectangle + corner label
         ],

         nodes: [
           {id, type, label, x, y, w, h, module?}
           //  type:  'tensor' (rounded rect) | 'function' (sharp rect)
           //  label: string OR string[]    (multi-line)
           //  module: id of containing module (omitted for inputs/outputs)
         ],

         edges: [
           [from_id, to_id, opts?]
           //  opts.path: explicit SVG path string (overrides auto-routing)
           //  opts.loop: true  → renders dashed AND excluded from
           //                     connectivity (so hover doesn't light up
           //                     the entire graph through the back-edge)
         ],

         plotLabels: [                    // optional
           {label, cx, cy}                // small blue badge with white letter
         ],

         moduleParents: {                 // optional, for nested modules
           child_id: parent_id            // hovering a node in `child_id`
         }                                // also highlights `parent_id`
       }

     Hover behavior:
       Hovering a node or edge highlights that node's strict ancestors
       and descendants in the (loop-edge-stripped) DAG. Modules whose
       nodes are in the highlighted set get their borders lit.

     Page convention:
       The HTML file just needs <svg id="d"></svg> and a <link> to
       diagram.css. SVG marker defs are injected automatically.
   ==================================================================== */
(function () {
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function ensureArrowDefs(svg) {
    let defs = svg.querySelector('defs');
    if (!defs) {
      defs = el('defs');
      svg.insertBefore(defs, svg.firstChild);
    }
    if (!defs.querySelector('#arrow')) {
      const m = el('marker', {
        id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse',
      });
      m.appendChild(el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#555' }));
      defs.appendChild(m);
    }
    if (!defs.querySelector('#arrow-hi')) {
      const m = el('marker', {
        id: 'arrow-hi', viewBox: '0 0 10 10', refX: 9, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse',
      });
      m.appendChild(el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#d2691e' }));
      defs.appendChild(m);
    }
  }

  function renderDiagram(svg, spec) {
    if (typeof svg === 'string') svg = document.querySelector(svg);
    if (!svg) throw new Error('renderDiagram: svg element not found');

    if (spec.width)  svg.setAttribute('width',  spec.width);
    if (spec.height) svg.setAttribute('height', spec.height);
    if (spec.width && spec.height) {
      svg.setAttribute('viewBox', '0 0 ' + spec.width + ' ' + spec.height);
    }

    ensureArrowDefs(svg);

    const modules       = spec.modules       || [];
    const nodes         = spec.nodes         || [];
    const edges         = spec.edges         || [];
    const plotLabels    = spec.plotLabels    || [];
    const moduleParents = spec.moduleParents || {};

    // ---- 1) modules (rendered first → behind nodes/edges) ----
    modules.forEach(m => {
      svg.appendChild(el('rect', {
        class: 'module', id: 'm_' + m.id,
        x: m.x, y: m.y, width: m.w, height: m.h,
      }));
      const t = el('text', {
        class: 'module-label', id: 'ml_' + m.id,
        x: m.x + 12, y: m.y + 18,
      });
      t.textContent = m.label;
      svg.appendChild(t);
    });

    // ---- 2) node lookup with derived edge anchors ----
    const nodeMap = {};
    nodes.forEach(n => {
      nodeMap[n.id] = n;
      n.cx     = n.x + n.w / 2;
      n.cy     = n.y + n.h / 2;
      n.right  = n.x + n.w;
      n.bottom = n.y + n.h;
    });

    function defaultPath(from, to) {
      const a = nodeMap[from], b = nodeMap[to];
      const sx = a.cx, sy = a.bottom;
      const tx = b.cx, ty = b.y;
      if (Math.abs(sx - tx) < 2) return 'M ' + sx + ' ' + sy + ' L ' + tx + ' ' + ty;
      const my = (sy + ty) / 2;
      return 'M ' + sx + ' ' + sy + ' L ' + sx + ' ' + my +
             ' L ' + tx + ' ' + my + ' L ' + tx + ' ' + ty;
    }

    // ---- 3) edges ----
    edges.forEach(([from, to, opts]) => {
      const o = opts || {};
      svg.appendChild(el('path', {
        class:        'edge' + (o.loop ? ' loop' : ''),
        d:            o.path || defaultPath(from, to),
        'marker-end': 'url(#arrow)',
        'data-from':  from,
        'data-to':    to,
      }));
    });

    // ---- 4) nodes (drawn last → on top of edges + modules) ----
    nodes.forEach(n => {
      const g = el('g', { class: 'node ' + n.type, 'data-id': n.id });
      g.appendChild(el('rect', { x: n.x, y: n.y, width: n.w, height: n.h }));
      const lines = Array.isArray(n.label) ? n.label : [n.label];
      lines.forEach((line, i) => {
        const t = el('text', {
          x: n.cx,
          y: n.y + n.h / 2 + (i - (lines.length - 1) / 2) * 16,
        });
        t.textContent = line;
        g.appendChild(t);
      });
      svg.appendChild(g);
    });

    // ---- 5) plot labels (above everything) ----
    plotLabels.forEach(p => {
      const g = el('g', { class: 'plot-label' });
      g.appendChild(el('circle', { cx: p.cx, cy: p.cy, r: 13 }));
      const t = el('text', { x: p.cx, y: p.cy });
      t.textContent = p.label;
      g.appendChild(t);
      svg.appendChild(g);
    });

    // ---- 6) hover: ancestors + descendants in the loop-stripped DAG ----
    const upstream = {}, downstream = {};
    edges.forEach(([from, to, opts]) => {
      if (opts && opts.loop) return;
      (downstream[from] = downstream[from] || []).push(to);
      (upstream[to]     = upstream[to]     || []).push(from);
    });

    function findConnected(nodeId) {
      const visited = new Set([nodeId]);
      const up = [nodeId];
      while (up.length) {
        const cur = up.pop();
        (upstream[cur] || []).forEach(n => {
          if (!visited.has(n)) { visited.add(n); up.push(n); }
        });
      }
      const down = [nodeId];
      while (down.length) {
        const cur = down.pop();
        (downstream[cur] || []).forEach(n => {
          if (!visited.has(n)) { visited.add(n); down.push(n); }
        });
      }
      return visited;
    }

    // Map each module to itself + any direct child modules (1-level nesting).
    const modulesByOwner = {};
    modules.forEach(m => { modulesByOwner[m.id] = [m.id]; });
    Object.entries(moduleParents).forEach(([child, parent]) => {
      if (modulesByOwner[parent]) modulesByOwner[parent].push(child);
    });

    function highlight(nodeId) {
      const visited = findConnected(nodeId);
      document.body.classList.add('hovering');

      svg.querySelectorAll('.node').forEach(e => {
        if (visited.has(e.dataset.id)) e.classList.add('highlighted');
      });
      svg.querySelectorAll('.edge').forEach(e => {
        if (visited.has(e.dataset.from) && visited.has(e.dataset.to)) {
          e.classList.add('highlighted');
          e.setAttribute('marker-end', 'url(#arrow-hi)');
        }
      });
      modules.forEach(m => {
        const owned = modulesByOwner[m.id];
        const has = nodes.some(n => owned.indexOf(n.module) !== -1 && visited.has(n.id));
        if (has) {
          document.getElementById('m_'  + m.id).classList.add('highlighted');
          document.getElementById('ml_' + m.id).classList.add('highlighted');
        }
      });
    }

    function clearHighlight() {
      document.body.classList.remove('hovering');
      document.querySelectorAll('.highlighted').forEach(e => e.classList.remove('highlighted'));
      svg.querySelectorAll('.edge').forEach(e => e.setAttribute('marker-end', 'url(#arrow)'));
    }

    svg.querySelectorAll('.node').forEach(e => {
      e.addEventListener('mouseenter', () => highlight(e.dataset.id));
      e.addEventListener('mouseleave', clearHighlight);
    });
    svg.querySelectorAll('.edge').forEach(e => {
      e.addEventListener('mouseenter', () => highlight(e.dataset.from));
      e.addEventListener('mouseleave', clearHighlight);
    });
  }

  window.renderDiagram = renderDiagram;
})();
