/* ============================================================
   GR00T-N1.5 tensor-flow diagram spec.
   Consumed by gr00t.html via:
     <script src="diagram_lib/diagram.js"></script>
     <script src="gr00t_data.js"></script>
     <script>renderDiagram('#d', gr00tSpec);</script>
   ============================================================ */
const gr00tSpec = {
  width:  1280,
  height: 1560,

  modules: [
    { id: 'vlm',        label: 'VLM (Eagle-2)',      x:  80, y: 130, w: 300, h:  560 },
    { id: 'vlpost',     label: 'VL post-processing', x:  80, y: 710, w: 300, h:  240 },
    { id: 'actionhead', label: 'Action head',        x: 610, y: 120, w: 570, h: 1360 },
    { id: 'dit',        label: 'DiT (sub-module)',   x: 740, y: 970, w: 320, h:  480 },
  ],

  nodes: [
    // ----- inputs (no module) -----
    { id: 'raw_frames',    type: 'tensor',   label: ['raw frames', '[B, V, 3, 224, 224]'], x: 130, y:  30, w: 200, h: 50 },
    { id: 'instruction',   type: 'tensor',   label: ['instruction', '[B, text_len]'],      x: 400, y:  30, w: 200, h: 50 },
    { id: 'state',         type: 'tensor',   label: ['state', '[B, 1, 64]'],               x: 640, y:  30, w: 160, h: 50 },
    { id: 'noisy_actions', type: 'tensor',   label: ['noisy actions', '[B, 40, 48]'],      x: 990, y:  30, w: 180, h: 50 },

    // ----- VLM -----
    { id: 'vision_model',  type: 'function', label: ['vision_model (SigLIP-2)'],           x: 130, y: 150, w: 200, h: 36, module: 'vlm' },
    { id: 't_siglip',      type: 'tensor',   label: ['[B*V, 256, 1152]'],                  x: 130, y: 200, w: 200, h: 32, module: 'vlm' },
    { id: 'pixel_shuffle', type: 'function', label: ['pixel_shuffle (sf=0.5)'],            x: 130, y: 246, w: 200, h: 36, module: 'vlm' },
    { id: 't_pix',         type: 'tensor',   label: ['[B*V, 64, 4608]'],                   x: 130, y: 296, w: 200, h: 32, module: 'vlm' },
    { id: 'mlp1',          type: 'function', label: ['mlp1'],                              x: 130, y: 342, w: 200, h: 36, module: 'vlm' },
    { id: 't_mlp',         type: 'tensor',   label: ['[B*V, 64, 2048]'],                   x: 130, y: 392, w: 200, h: 32, module: 'vlm' },
    { id: 'splice',        type: 'function', label: ['splice_into_LLM'],                   x: 130, y: 438, w: 200, h: 36, module: 'vlm' },
    { id: 't_combined',    type: 'tensor',   label: ['[B, T_vl, 2048]'],                   x: 130, y: 488, w: 200, h: 32, module: 'vlm' },
    { id: 'qwen3',         type: 'function', label: ['Qwen3 (layers 0-11)'],               x: 130, y: 534, w: 200, h: 36, module: 'vlm' },
    { id: 'vl_embs',       type: 'tensor',   label: ['vl_embs', '[B, T_vl, 2048]'],        x: 130, y: 584, w: 200, h: 50, module: 'vlm' },

    // ----- VL post -----
    { id: 'vlln',          type: 'function', label: ['vlln (LayerNorm)'],                  x: 130, y: 730, w: 200, h: 36, module: 'vlpost' },
    { id: 't_lln',         type: 'tensor',   label: ['[B, T_vl, 2048]'],                   x: 130, y: 776, w: 200, h: 32, module: 'vlpost' },
    { id: 'vl_self_attn',  type: 'function', label: ['vl_self_attention (4 layers)'],      x: 130, y: 818, w: 200, h: 36, module: 'vlpost' },
    { id: 'vl_embs_post',  type: 'tensor',   label: ["vl_embs'", '[B, T_vl, 2048]'],       x: 130, y: 864, w: 200, h: 50, module: 'vlpost' },

    // ----- Action head: 3 parallel sub-streams converging into concat -----
    { id: 'state_encoder',   type: 'function', label: ['state_encoder'],                     x:  640, y: 150, w: 160, h: 36, module: 'actionhead' },
    { id: 'state_features',  type: 'tensor',   label: ['state_features', '[B, 1, 1536]'],    x:  640, y: 200, w: 160, h: 50, module: 'actionhead' },

    { id: 'future_tokens',   type: 'function', label: ['future_tokens (Embedding)'],         x:  820, y: 150, w: 180, h: 36, module: 'actionhead' },
    { id: 'future_t',        type: 'tensor',   label: ['future_t', '[B, 32, 1536]'],         x:  820, y: 200, w: 180, h: 50, module: 'actionhead' },

    { id: 'action_encoder',  type: 'function', label: ['action_encoder'],                    x: 1010, y: 150, w: 140, h: 36, module: 'actionhead' },
    { id: 'action_features', type: 'tensor',   label: ['action_features', '[B, 40, 1536]'],  x: 1010, y: 200, w: 140, h: 50, module: 'actionhead' },

    { id: 'concat',          type: 'function', label: ['concat'],                            x:  780, y: 280, w: 240, h: 36, module: 'actionhead' },
    { id: 'sa_embs',         type: 'tensor',   label: ['sa_embs', '[B, 73, 1536]'],          x:  780, y: 330, w: 240, h: 50, module: 'actionhead' },

    // ----- DiT (sub-module of Action head) -----
    { id: 'dit_block',     type: 'function', label: ['DiT 16 layers', '(alternating cross / self-attn)'], x: 760, y:  990, w: 280, h: 50, module: 'dit' },
    { id: 't_dit_out',     type: 'tensor',   label: ['[B, 73, 1536]'],                                    x: 760, y: 1054, w: 280, h: 32, module: 'dit' },
    { id: 'adaln_linear',  type: 'function', label: ['AdaLN + Linear(1536 → 1024)'],                      x: 760, y: 1100, w: 280, h: 36, module: 'dit' },
    { id: 'model_output',  type: 'tensor',   label: ['model_output', '[B, 73, 1024]'],                    x: 760, y: 1150, w: 280, h: 50, module: 'dit' },
    { id: 'slice_action',  type: 'function', label: ['slice [:, 33:73, :]'],                              x: 760, y: 1214, w: 280, h: 36, module: 'dit' },
    { id: 't_slice',       type: 'tensor',   label: ['[B, 40, 1024]'],                                    x: 760, y: 1264, w: 280, h: 32, module: 'dit' },
    { id: 'action_decoder',type: 'function', label: ['action_decoder'],                                   x: 760, y: 1310, w: 280, h: 36, module: 'dit' },
    { id: 'velocity',      type: 'tensor',   label: ['velocity', '[B, 40, 48]'],                          x: 760, y: 1360, w: 280, h: 50, module: 'dit' },

    // ----- output (outside Action head) -----
    { id: 'action_pred',   type: 'tensor',   label: ['action_pred', '[B, 40, 48]'], x: 800, y: 1500, w: 200, h: 50 },
  ],

  edges: [
    // inputs into modules
    ['raw_frames',  'vision_model'],
    ['instruction', 'splice', { path: 'M 500 80 L 500 456 L 330 456' }],

    // VLM
    ['vision_model',  't_siglip'],
    ['t_siglip',      'pixel_shuffle'],
    ['pixel_shuffle', 't_pix'],
    ['t_pix',         'mlp1'],
    ['mlp1',          't_mlp'],
    ['t_mlp',         'splice'],
    ['splice',        't_combined'],
    ['t_combined',    'qwen3'],
    ['qwen3',         'vl_embs'],

    // VL post
    ['vl_embs',     'vlln'],
    ['vlln',        't_lln'],
    ['t_lln',       'vl_self_attn'],

    // Action head
    ['state',           'state_encoder'],
    ['state_encoder',   'state_features'],
    ['state_features',  'concat'],
    ['future_tokens',   'future_t'],
    ['future_t',        'concat'],
    ['noisy_actions',   'action_encoder'],
    ['action_encoder',  'action_features'],
    ['action_features', 'concat'],
    ['concat',          'sa_embs'],

    // VL post output → DiT
    ['vl_self_attn', 'vl_embs_post'],
    ['vl_embs_post', 'dit_block'],
    ['sa_embs',      'dit_block'],

    // DiT
    ['dit_block',      't_dit_out'],
    ['t_dit_out',      'adaln_linear'],
    ['adaln_linear',   'model_output'],
    ['model_output',   'slice_action'],
    ['slice_action',   't_slice'],
    ['t_slice',        'action_decoder'],
    ['action_decoder', 'velocity'],

    // output
    ['velocity', 'action_pred'],

    // loop back to noisy_actions: route around the right side of the canvas
    ['velocity', 'noisy_actions',
      { loop: true, path: 'M 1040 1385 L 1240 1385 L 1240 110 L 1080 110 L 1080 80' }],
  ],

  // Plot labels:
  //   A → action_pred         (final chunk, plotted by ChunkPanel)
  //   B → noisy_actions       (denoising trajectory captured at action_encoder hook)
  //   C → dit_block (top)     (cross-attn weights, even-indexed DiT layers)
  //   D → dit_block (bottom)  (self-attn weights,  odd-indexed DiT layers)
  plotLabels: [
    { label: 'A', cx: 1020, cy: 1525 },
    { label: 'B', cx: 1200, cy:   55 },
    { label: 'C', cx: 1070, cy: 1000 },
    { label: 'D', cx: 1070, cy: 1032 },
  ],

  // DiT is rendered as a sub-module of Action head; hovering any DiT node
  // also lights up the outer Action head rectangle.
  moduleParents: {
    dit: 'actionhead',
  },
};
