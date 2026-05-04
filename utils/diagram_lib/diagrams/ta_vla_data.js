/* ============================================================
   TA-VLA tensor-flow diagram spec.
   Built from /Users/youngwoong/workspace/TA-VLA.

   Shown variant:  EffortType.EXPERT_HIS_C_FUT
     - decoder-side  injection (effort token enters the action expert)
     - history-concat (single token built from |hist|-frame torque history)
     - joint future-effort prediction (action_in/out_proj widen by effort_dim,
       loss = action_loss + 0.1 · effort_loss)

   Reference defaults
     (TA-VLA/src/openpi/models/pi0.py, shared/effort_type.py):
       action_dim     = 32
       action_horizon = 50
       max_token_len  = 48
       effort_dim     = 14
       effort_history = (offset_1, ..., offset_n)    |hist| ≡ n_hist
       paligemma  = gemma_2b   (W=2048, D=18, MLP=16384, MQA-8h, head=256)
       act_expert = gemma_300m (W=1024, D=18, MLP=4096,  MQA-8h, head=256)
   ============================================================ */
const taVlaSpec = {
  width:  1700,
  height: 1750,

  modules: [
    { id: 'siglip',     label: 'SigLIP-So400m/14  (per-camera)',                    x:  30, y: 120, w:  320, h: 480 },
    { id: 'vlm_embed',  label: 'PaliGemma embedder',                                x: 350, y: 120, w:  180, h: 130 },
    { id: 'effort_br',  label: 'Effort projector',                                  x: 530, y: 120, w:  200, h: 320 },
    { id: 'suffix_enc', label: 'Suffix encoder',                                    x: 750, y: 120, w:  620, h: 460 },
    { id: 'gemma',      label: 'Joint Gemma  (PaliGemma 2B + Action Expert 300M)',  x:  30, y: 820, w: 1640, h: 420 },
  ],

  nodes: [
    // ----- inputs -----
    { id: 'images',        type: 'tensor',   label: ['images',        '[B*3, 224, 224, 3]'], x:  70, y:  30, w: 240, h: 50 },
    { id: 'prompt',        type: 'tensor',   label: ['prompt',        '[B, 48]'],            x: 350, y:  30, w: 180, h: 50 },
    { id: 'effort',        type: 'tensor',   label: ['effort',        '[B, n_hist, 14]'],    x: 550, y:  30, w: 180, h: 50 },
    { id: 'state',         type: 'tensor',   label: ['state',         '[B, 32]'],            x: 770, y:  30, w: 140, h: 50 },
    { id: 'noisy_actions', type: 'tensor',   label: ['noisy_actions', '[B, 50, 32+14]'],     x: 940, y:  30, w: 230, h: 50 },
    { id: 'timestep',      type: 'tensor',   label: ['timestep',      '[B]'],                x:1200, y:  30, w: 140, h: 50 },

    // ----- SigLIP vision encoder -----
    { id: 'patch_embed',   type: 'function', label: ['Conv 14×14, stride 14'],               x:  50, y: 140, w: 280, h: 36, module: 'siglip' },
    { id: 't_patch',       type: 'tensor',   label: ['[B*3, 256, 1152]'],                    x:  50, y: 180, w: 280, h: 32, module: 'siglip' },
    { id: 'sincos2d',      type: 'function', label: ['+ posemb_sincos2d'],                   x:  50, y: 218, w: 280, h: 36, module: 'siglip' },
    { id: 't_pe',          type: 'tensor',   label: ['[B*3, 256, 1152]'],                    x:  50, y: 258, w: 280, h: 32, module: 'siglip' },
    { id: 'siglip_enc',    type: 'function', label: ['27× Encoder1DBlock', '(LN + MHSA-16h + MLP-4304)'], x:  50, y: 296, w: 280, h: 50, module: 'siglip' },
    { id: 't_siglip_enc',  type: 'tensor',   label: ['[B*3, 256, 1152]'],                    x:  50, y: 350, w: 280, h: 32, module: 'siglip' },
    { id: 'siglip_ln',     type: 'function', label: ['encoder_norm  (LayerNorm)'],           x:  50, y: 390, w: 280, h: 36, module: 'siglip' },
    { id: 'siglip_head',   type: 'function', label: ['head Dense  (1152 → 2048)'],           x:  50, y: 430, w: 280, h: 36, module: 'siglip' },
    { id: 't_img_per_cam', type: 'tensor',   label: ['[B*3, 256, 2048]'],                    x:  50, y: 470, w: 280, h: 32, module: 'siglip' },
    { id: 'cam_concat',    type: 'function', label: ['rearrange  (B V) S D → B (V S) D'],    x:  50, y: 506, w: 280, h: 36, module: 'siglip' },
    { id: 'image_tokens',  type: 'tensor',   label: ['image_tokens', '[B, 768, 2048]'],      x:  50, y: 546, w: 280, h: 50, module: 'siglip' },

    // ----- PaliGemma embedder (language) -----
    { id: 'embed_lookup',  type: 'function', label: ['embedder.encode  (× √2048)'],          x: 360, y: 140, w: 160, h: 36, module: 'vlm_embed' },
    { id: 'lang_tokens',   type: 'tensor',   label: ['lang_tokens', '[B, 48, 2048]'],        x: 360, y: 180, w: 160, h: 50, module: 'vlm_embed' },

    // ----- Effort projector -----
    // d_in = 14 · n_hist  (history flattened along the time axis)
    // 2W = 2048, W = 1024  (decoder-side widths)
    { id: 'eff_flatten',   type: 'function', label: ['flatten history'],                     x: 550, y: 140, w: 160, h: 36, module: 'effort_br' },
    { id: 't_eff_flat',    type: 'tensor',   label: ['[B, 14·n_hist]'],                      x: 550, y: 180, w: 160, h: 32, module: 'effort_br' },
    { id: 'eff_proj_in',   type: 'function', label: ['effort_proj_in', '(14·n_hist → 2048)'],x: 550, y: 218, w: 160, h: 50, module: 'effort_br' },
    { id: 'eff_swish',     type: 'function', label: ['swish'],                               x: 550, y: 275, w: 160, h: 36, module: 'effort_br' },
    { id: 'eff_proj_out',  type: 'function', label: ['effort_proj_out', '(2048 → 1024)'],    x: 550, y: 315, w: 160, h: 50, module: 'effort_br' },
    { id: 'effort_token',  type: 'tensor',   label: ['effort_token', '[B, 1, 1024]'],        x: 550, y: 375, w: 160, h: 50, module: 'effort_br' },

    // ----- Suffix encoder -----
    { id: 'state_proj',         type: 'function', label: ['state_proj  (32 → 1024)'],        x: 770, y: 140, w: 140, h: 36, module: 'suffix_enc' },
    { id: 'state_token',        type: 'tensor',   label: ['state_token', '[B, 1, 1024]'],    x: 770, y: 180, w: 140, h: 50, module: 'suffix_enc' },

    { id: 'action_in_proj',     type: 'function', label: ['action_in_proj', '(32+14 → 1024)'], x: 940, y: 140, w: 220, h: 50, module: 'suffix_enc' },
    { id: 'action_tokens',      type: 'tensor',   label: ['action_tokens', '[B, 50, 1024]'],   x: 940, y: 200, w: 220, h: 50, module: 'suffix_enc' },

    { id: 'time_sincos',        type: 'function', label: ['posemb_sincos', '(D=1024)'],        x:1190, y: 140, w: 170, h: 50, module: 'suffix_enc' },
    { id: 'time_emb',           type: 'tensor',   label: ['time_emb', '[B, 1024]'],            x:1190, y: 200, w: 170, h: 50, module: 'suffix_enc' },
    { id: 'time_repeat',        type: 'function', label: ['repeat  s=50'],                     x:1190, y: 260, w: 170, h: 36, module: 'suffix_enc' },
    { id: 'time_tokens',        type: 'tensor',   label: ['[B, 50, 1024]'],                    x:1190, y: 300, w: 170, h: 32, module: 'suffix_enc' },

    { id: 'at_concat',          type: 'function', label: ['concat  (axis=-1)'],                x: 940, y: 350, w: 420, h: 36, module: 'suffix_enc' },
    { id: 't_action_time_2048', type: 'tensor',   label: ['[B, 50, 2048]'],                    x: 940, y: 390, w: 420, h: 32, module: 'suffix_enc' },
    { id: 'at_mlp_in',          type: 'function', label: ['action_time_mlp_in  (2048 → 1024)  + swish'], x: 940, y: 425, w: 420, h: 36, module: 'suffix_enc' },
    { id: 'at_mlp_out',         type: 'function', label: ['action_time_mlp_out  (1024 → 1024)'],         x: 940, y: 465, w: 420, h: 36, module: 'suffix_enc' },
    { id: 'action_time_tokens', type: 'tensor',   label: ['action_time_tokens', '[B, 50, 1024]'],         x: 940, y: 505, w: 420, h: 50, module: 'suffix_enc' },

    // ----- Concats outside modules -----
    { id: 'prefix_concat', type: 'function', label: ['concat  (axis=1)'],                                                          x: 150, y: 640, w: 320, h: 50 },
    { id: 'prefix_tokens', type: 'tensor',   label: ['prefix_tokens', '[B, 816, 2048]'],                                            x: 150, y: 720, w: 320, h: 50 },

    { id: 'suffix_concat', type: 'function', label: ['concat  (axis=1)', 'effort_token  ·  state_token  ·  action_time_tokens'],   x: 720, y: 640, w: 620, h: 50 },
    { id: 'suffix_tokens', type: 'tensor',   label: ['suffix_tokens', '[B, 1 + 1 + 50, 1024]  =  [B, 52, 1024]'],                  x: 870, y: 720, w: 320, h: 50 },

    // ----- Joint Gemma transformer -----
    { id: 'pre_norm_p',  type: 'function', label: ['pre_attention_norm  (RMSNorm)'],     x: 180, y: 860, w: 320, h: 36, module: 'gemma' },
    { id: 'pre_norm_s',  type: 'function', label: ['pre_attention_norm  (RMSNorm)'],     x:1180, y: 860, w: 320, h: 36, module: 'gemma' },

    { id: 'joint_block', type: 'function', label: ['DualBlock  ×18 layers',
                                                   'joint MoE attention   (concat K,V across experts)',
                                                   'per-expert RMSNorm  +  GeGLU FFN  +  residuals'],
                                                  x: 580, y: 920, w: 540, h: 180, module: 'gemma' },

    { id: 'final_norm_p', type: 'function', label: ['final_norm  (RMSNorm)'],            x: 180, y: 1120, w: 320, h: 36, module: 'gemma' },
    { id: 'prefix_out',   type: 'tensor',   label: ['prefix_out  (unused)'],             x: 180, y: 1160, w: 320, h: 50, module: 'gemma' },

    { id: 'final_norm_s', type: 'function', label: ['final_norm  (RMSNorm)'],            x:1180, y: 1120, w: 320, h: 36, module: 'gemma' },
    { id: 'suffix_out',   type: 'tensor',   label: ['suffix_out', '[B, 52, 1024]'],      x:1180, y: 1160, w: 320, h: 50, module: 'gemma' },

    // ----- Output / sampler -----
    { id: 'slice_action',    type: 'function', label: ['slice  [:, -50:, :]'],                                       x:1180, y: 1260, w: 320, h: 50 },
    { id: 'action_out_proj', type: 'function', label: ['action_out_proj', '(1024 → 32+14)'],                         x:1180, y: 1320, w: 320, h: 50 },
    { id: 'velocity',        type: 'tensor',   label: ['velocity v_t', '[B, 50, 32+14]'],                            x:1180, y: 1380, w: 320, h: 50 },
    { id: 'euler',           type: 'function', label: ['x_t  ←  x_t  +  dt · v_t', 'dt = −1 / num_steps   (=10)'],   x:1180, y: 1450, w: 320, h: 50 },
    { id: 'action_pred',     type: 'tensor',   label: ['action_pred  =  x_0[..., :32]', '[B, 50, 32]'],              x:1180, y: 1530, w: 320, h: 50 },

    // ----- Training-time loss (sibling to euler) -----
    { id: 'loss_box',        type: 'function', label: ['(training)  flow-matching MSE loss',
                                                       'action_loss    =  ‖v_t[..., :32]  −  u_t[..., :32]‖²',
                                                       'effort_loss   =  ‖v_t[..., 32:]  −  u_t[..., 32:]‖²',
                                                       'loss = action_loss  +  0.1 · effort_loss'],
                                                  x: 660, y: 1440, w: 460, h: 100 },
  ],

  edges: [
    // inputs to encoders
    ['images',        'patch_embed'],
    ['prompt',        'embed_lookup'],
    ['effort',        'eff_flatten'],
    ['state',         'state_proj'],
    ['noisy_actions', 'action_in_proj'],
    ['timestep',      'time_sincos'],

    // SigLIP
    ['patch_embed',   't_patch'],
    ['t_patch',       'sincos2d'],
    ['sincos2d',      't_pe'],
    ['t_pe',          'siglip_enc'],
    ['siglip_enc',    't_siglip_enc'],
    ['t_siglip_enc',  'siglip_ln'],
    ['siglip_ln',     'siglip_head'],
    ['siglip_head',   't_img_per_cam'],
    ['t_img_per_cam', 'cam_concat'],
    ['cam_concat',    'image_tokens'],

    // VLM embedder
    ['embed_lookup', 'lang_tokens'],

    // Effort projector
    ['eff_flatten',  't_eff_flat'],
    ['t_eff_flat',   'eff_proj_in'],
    ['eff_proj_in',  'eff_swish'],
    ['eff_swish',    'eff_proj_out'],
    ['eff_proj_out', 'effort_token'],

    // Prefix concat (image + lang only — effort is decoder-side)
    ['image_tokens', 'prefix_concat'],
    // route lang_tokens around the right side of SigLIP into prefix_concat
    ['lang_tokens',  'prefix_concat',
      { path: 'M 440 230 L 440 615 L 320 615 L 320 640' }],
    ['prefix_concat', 'prefix_tokens'],

    // Suffix encoder internals
    ['state_proj',         'state_token'],
    ['action_in_proj',     'action_tokens'],
    ['time_sincos',        'time_emb'],
    ['time_emb',           'time_repeat'],
    ['time_repeat',        'time_tokens'],
    ['action_tokens',      'at_concat'],
    ['time_tokens',        'at_concat'],
    ['at_concat',          't_action_time_2048'],
    ['t_action_time_2048', 'at_mlp_in'],
    ['at_mlp_in',          'at_mlp_out'],
    ['at_mlp_out',         'action_time_tokens'],

    // Suffix concat: effort_token | state_token | action_time_tokens
    ['effort_token', 'suffix_concat',
      { path: 'M 630 425 L 630 605 L 800 605 L 800 640' }],
    ['state_token',  'suffix_concat',
      { path: 'M 840 230 L 840 605 L 950 605 L 950 640' }],
    ['action_time_tokens', 'suffix_concat'],
    ['suffix_concat', 'suffix_tokens'],

    // Into joint Gemma
    ['prefix_tokens', 'pre_norm_p'],
    ['suffix_tokens', 'pre_norm_s'],

    // Joint Gemma flow (collapsed)
    ['pre_norm_p',   'joint_block'],
    ['pre_norm_s',   'joint_block'],
    ['joint_block',  'final_norm_p'],
    ['joint_block',  'final_norm_s'],
    ['final_norm_p', 'prefix_out'],
    ['final_norm_s', 'suffix_out'],

    // Output / sampler
    ['suffix_out',     'slice_action'],
    ['slice_action',   'action_out_proj'],
    ['action_out_proj','velocity'],
    ['velocity',       'euler'],
    ['euler',          'action_pred'],

    // Training-time loss branch
    ['velocity', 'loss_box',
      { path: 'M 1180 1405 L 1140 1405 L 1140 1440 L 1120 1440' }],

    // Sampling loop: euler step → noisy_actions
    ['euler', 'noisy_actions',
      { loop: true, path: 'M 1500 1475 L 1640 1475 L 1640 55 L 1170 55' }],
  ],
};
