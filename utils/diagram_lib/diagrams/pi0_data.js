/* ============================================================
   pi0 (openpi) tensor-flow diagram spec.
   Consumed by pi0.html via:
     <script src="diagram_lib/diagram.js"></script>
     <script src="pi0_data.js"></script>
     <script>renderDiagram('#d', pi0Spec);</script>

   Reference defaults (openpi/src/openpi/models/pi0_config.py):
     action_dim     = 32
     action_horizon = 50
     max_token_len  = 48     (pi0; pi05 = 200)
     paligemma_variant     = "gemma_2b"   (W=2048, D=18, MLP=16384, 8h MQA, head=256)
     action_expert_variant = "gemma_300m" (W=1024, D=18, MLP=4096,  8h MQA, head=256)
     vision encoder = SigLIP "So400m/14"  (W=1152, D=27, MLP=4304, 16h, patch=14)
   3 cameras: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb @ 224x224.
   ============================================================ */
const pi0Spec = {
  width:  1500,
  height: 1900,

  modules: [
    { id: 'siglip',     label: 'SigLIP-So400m/14  (per-camera, weight-shared)', x:  40, y: 120, w:  320, h: 480 },
    { id: 'vlm_embed',  label: 'PaliGemma embedder',                            x: 380, y: 120, w:  180, h: 130 },
    { id: 'suffix_enc', label: 'Suffix encoder',                                x: 600, y: 120, w:  820, h: 460 },
    { id: 'gemma',      label: 'Joint Gemma  (PaliGemma 2B + Action Expert 300M)', x:  40, y: 820, w: 1420, h: 740 },
    { id: 'dualblock',  label: 'DualBlock  ×18 layers',                         x: 100, y: 900, w: 1300, h: 540 },
  ],

  nodes: [
    // ----- inputs (no module) -----
    { id: 'images',        type: 'tensor',   label: ['images',        '[B*3, 224, 224, 3]'], x:  80, y:  30, w: 240, h: 50 },
    { id: 'prompt',        type: 'tensor',   label: ['prompt',        '[B, 48]'],            x: 380, y:  30, w: 180, h: 50 },
    { id: 'state',         type: 'tensor',   label: ['state',         '[B, 32]'],            x: 620, y:  30, w: 140, h: 50 },
    { id: 'noisy_actions', type: 'tensor',   label: ['noisy_actions', '[B, 50, 32]'],        x: 820, y:  30, w: 200, h: 50 },
    { id: 'timestep',      type: 'tensor',   label: ['timestep',      '[B]'],                x:1080, y:  30, w: 130, h: 50 },

    // ----- SigLIP vision encoder -----
    { id: 'patch_embed',   type: 'function', label: ['Conv 14×14, stride 14'],               x:  60, y: 140, w: 280, h: 36, module: 'siglip' },
    { id: 't_patch',       type: 'tensor',   label: ['[B*3, 256, 1152]'],                    x:  60, y: 180, w: 280, h: 32, module: 'siglip' },
    { id: 'sincos2d',      type: 'function', label: ['+ posemb_sincos2d'],                   x:  60, y: 218, w: 280, h: 36, module: 'siglip' },
    { id: 't_pe',          type: 'tensor',   label: ['[B*3, 256, 1152]'],                    x:  60, y: 258, w: 280, h: 32, module: 'siglip' },
    { id: 'siglip_enc',    type: 'function', label: ['27× Encoder1DBlock', '(LN + MHSA-16h + MLP-4304)'], x: 60, y: 296, w: 280, h: 50, module: 'siglip' },
    { id: 't_siglip_enc',  type: 'tensor',   label: ['[B*3, 256, 1152]'],                    x:  60, y: 350, w: 280, h: 32, module: 'siglip' },
    { id: 'siglip_ln',     type: 'function', label: ['encoder_norm  (LayerNorm)'],           x:  60, y: 390, w: 280, h: 36, module: 'siglip' },
    { id: 'siglip_head',   type: 'function', label: ['head Dense  (1152 → 2048)'],           x:  60, y: 430, w: 280, h: 36, module: 'siglip' },
    { id: 't_img_per_cam', type: 'tensor',   label: ['[B*3, 256, 2048]'],                    x:  60, y: 470, w: 280, h: 32, module: 'siglip' },
    { id: 'cam_concat',    type: 'function', label: ['rearrange  (B V) S D → B (V S) D'],    x:  60, y: 506, w: 280, h: 36, module: 'siglip' },
    { id: 'image_tokens',  type: 'tensor',   label: ['image_tokens', '[B, 768, 2048]'],      x:  60, y: 546, w: 280, h: 50, module: 'siglip' },

    // ----- PaliGemma embedder (language) -----
    { id: 'embed_lookup',  type: 'function', label: ['embedder.encode  (× √2048)'],          x: 390, y: 140, w: 160, h: 36, module: 'vlm_embed' },
    { id: 'lang_tokens',   type: 'tensor',   label: ['lang_tokens', '[B, 48, 2048]'],        x: 390, y: 180, w: 160, h: 50, module: 'vlm_embed' },

    // ----- Suffix encoder -----
    // state column
    { id: 'state_proj',         type: 'function', label: ['state_proj  (32 → 1024)'],        x: 620, y: 140, w: 160, h: 36, module: 'suffix_enc' },
    { id: 'state_token',        type: 'tensor',   label: ['state_token', '[B, 1, 1024]'],    x: 620, y: 180, w: 160, h: 50, module: 'suffix_enc' },

    // action column
    { id: 'action_in_proj',     type: 'function', label: ['action_in_proj  (32 → 1024)'],    x: 800, y: 140, w: 220, h: 36, module: 'suffix_enc' },
    { id: 'action_tokens',      type: 'tensor',   label: ['action_tokens', '[B, 50, 1024]'], x: 800, y: 180, w: 220, h: 50, module: 'suffix_enc' },

    // time column
    { id: 'time_sincos',        type: 'function', label: ['posemb_sincos', '(D=1024,  T=[4e-3, 4.0])'], x:1040, y: 140, w: 200, h: 50, module: 'suffix_enc' },
    { id: 'time_emb',           type: 'tensor',   label: ['time_emb', '[B, 1024]'],          x:1040, y: 200, w: 200, h: 50, module: 'suffix_enc' },
    { id: 'time_repeat',        type: 'function', label: ['repeat  s=50'],                   x:1040, y: 260, w: 200, h: 36, module: 'suffix_enc' },
    { id: 'time_tokens',        type: 'tensor',   label: ['[B, 50, 1024]'],                  x:1040, y: 300, w: 200, h: 32, module: 'suffix_enc' },

    // action+time mixing
    { id: 'at_concat',          type: 'function', label: ['concat  (axis=-1)'],              x: 800, y: 350, w: 440, h: 36, module: 'suffix_enc' },
    { id: 't_action_time_2048', type: 'tensor',   label: ['[B, 50, 2048]'],                  x: 800, y: 390, w: 440, h: 32, module: 'suffix_enc' },
    { id: 'at_mlp_in',          type: 'function', label: ['action_time_mlp_in  (2048 → 1024)  + swish'], x: 800, y: 425, w: 440, h: 36, module: 'suffix_enc' },
    { id: 'at_mlp_out',         type: 'function', label: ['action_time_mlp_out  (1024 → 1024)'],         x: 800, y: 465, w: 440, h: 36, module: 'suffix_enc' },
    { id: 'action_time_tokens', type: 'tensor',   label: ['action_time_tokens', '[B, 50, 1024]'],         x: 800, y: 505, w: 440, h: 50, module: 'suffix_enc' },

    // ----- prefix concat (top-level) -----
    { id: 'prefix_concat', type: 'function', label: ['concat  (axis=1)'],                    x: 200, y: 640, w: 280, h: 36 },
    { id: 'prefix_tokens', type: 'tensor',   label: ['prefix_tokens', '[B, 816, 2048]'],     x: 200, y: 720, w: 280, h: 50 },

    // ----- suffix concat (top-level) -----
    { id: 'suffix_concat', type: 'function', label: ['concat  (axis=1)'],                    x: 720, y: 640, w: 480, h: 36 },
    { id: 'suffix_tokens', type: 'tensor',   label: ['suffix_tokens', '[B, 51, 1024]'],      x: 760, y: 720, w: 400, h: 50 },

    // ----- Joint Gemma · DualBlock (one of 18 layers) -----
    // Prefix expert column (PaliGemma 2B), left
    { id: 'pre_norm_p',    type: 'function', label: ['pre_attention_norm  (RMSNorm)'],       x: 140, y: 920, w: 320, h: 36, module: 'dualblock' },
    { id: 'qkv_p',         type: 'function', label: ['q,k,v einsum',  'BSD,3KDH→3BSKH   (D=2048)'],   x: 140, y: 960, w: 320, h: 50, module: 'dualblock' },

    // Suffix expert column (Action Expert 300M), right
    { id: 'pre_norm_s',    type: 'function', label: ['pre_attention_norm  (RMSNorm)'],       x:1040, y: 920, w: 320, h: 36, module: 'dualblock' },
    { id: 'qkv_s',         type: 'function', label: ['q,k,v einsum',  'BSD,3KDH→3BSKH   (D=1024)'],   x:1040, y: 960, w: 320, h: 50, module: 'dualblock' },

    // Joint attention center
    { id: 'attn_concat',   type: 'function', label: ['concat q,k,v across experts  (axis=1)'], x: 540, y: 1030, w: 420, h: 36, module: 'dualblock' },
    { id: 'rope',          type: 'function', label: ['RoPE  on q, k'],                       x: 540, y: 1070, w: 420, h: 36, module: 'dualblock' },
    { id: 'sdpa',          type: 'function', label: ['softmax( QKᵀ / √H + mask ) · V', '(block-causal:  prefix↔prefix,  suffix → all)'], x: 540, y: 1110, w: 420, h: 50, module: 'dualblock' },
    { id: 'split_attn',    type: 'function', label: ['split by expert'],                     x: 540, y: 1165, w: 420, h: 36, module: 'dualblock' },

    // Continued prefix flow
    { id: 'out_p',         type: 'function', label: ['attn_out  (NHD → 2048)'],              x: 140, y: 1230, w: 320, h: 36, module: 'dualblock' },
    { id: 'res1_p',        type: 'function', label: ['+  residual'],                         x: 140, y: 1270, w: 320, h: 36, module: 'dualblock' },
    { id: 'ffn_p',         type: 'function', label: ['pre_ffw_norm  +  GeGLU FFN', '(2048 → 16384 → 2048)'], x: 140, y: 1310, w: 320, h: 50, module: 'dualblock' },
    { id: 'res2_p',        type: 'function', label: ['+  residual'],                         x: 140, y: 1364, w: 320, h: 36, module: 'dualblock' },

    // Continued suffix flow
    { id: 'out_s',         type: 'function', label: ['attn_out  (NHD → 1024)'],              x:1040, y: 1230, w: 320, h: 36, module: 'dualblock' },
    { id: 'res1_s',        type: 'function', label: ['+  residual'],                         x:1040, y: 1270, w: 320, h: 36, module: 'dualblock' },
    { id: 'ffn_s',         type: 'function', label: ['pre_ffw_norm  +  GeGLU FFN', '(1024 → 4096 → 1024)'],  x:1040, y: 1310, w: 320, h: 50, module: 'dualblock' },
    { id: 'res2_s',        type: 'function', label: ['+  residual'],                         x:1040, y: 1364, w: 320, h: 36, module: 'dualblock' },

    // ----- After 18 layers (inside gemma, outside dualblock) -----
    { id: 'final_norm_p',  type: 'function', label: ['final_norm  (RMSNorm)'],               x: 140, y: 1460, w: 320, h: 36, module: 'gemma' },
    { id: 'prefix_out',    type: 'tensor',   label: ['prefix_out', '[B, 816, 2048]'],        x: 140, y: 1500, w: 320, h: 50, module: 'gemma' },

    { id: 'final_norm_s',  type: 'function', label: ['final_norm  (RMSNorm)'],               x:1040, y: 1460, w: 320, h: 36, module: 'gemma' },
    { id: 'suffix_out',    type: 'tensor',   label: ['suffix_out', '[B, 51, 1024]'],         x:1040, y: 1500, w: 320, h: 50, module: 'gemma' },

    // ----- Output / flow-matching sampler -----
    { id: 'slice_action',  type: 'function', label: ['slice  [:, -50:, :]'],                 x:1040, y: 1590, w: 320, h: 36 },
    { id: 'action_out_proj', type: 'function', label: ['action_out_proj  (1024 → 32)'],      x:1040, y: 1630, w: 320, h: 36 },
    { id: 'velocity',      type: 'tensor',   label: ['velocity v_t', '[B, 50, 32]'],         x:1040, y: 1670, w: 320, h: 50 },
    { id: 'euler',         type: 'function', label: ['x_t  ←  x_t  +  dt · v_t', '(dt = −1 / num_steps,    num_steps = 10)'], x:1040, y: 1735, w: 320, h: 50 },
    { id: 'action_pred',   type: 'tensor',   label: ['action_pred  =  x_0', '[B, 50, 32]'],  x:1040, y: 1810, w: 320, h: 50 },
  ],

  edges: [
    // ----- inputs into pre-encoders -----
    ['images',        'patch_embed'],
    ['prompt',        'embed_lookup'],
    ['state',         'state_proj'],
    ['noisy_actions', 'action_in_proj'],
    ['timestep',      'time_sincos'],

    // ----- SigLIP -----
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

    // ----- Language embedder -----
    ['embed_lookup',  'lang_tokens'],

    // ----- Prefix concat -----
    ['image_tokens',  'prefix_concat'],
    // route lang_tokens around the right side of the SigLIP module instead of through it
    ['lang_tokens',   'prefix_concat',
      { path: 'M 470 230 L 470 615 L 340 615 L 340 640' }],
    ['prefix_concat', 'prefix_tokens'],

    // ----- Suffix encoder internals -----
    ['state_proj',          'state_token'],
    ['action_in_proj',      'action_tokens'],
    ['time_sincos',         'time_emb'],
    ['time_emb',            'time_repeat'],
    ['time_repeat',         'time_tokens'],
    ['action_tokens',       'at_concat'],
    ['time_tokens',         'at_concat'],
    ['at_concat',           't_action_time_2048'],
    ['t_action_time_2048',  'at_mlp_in'],
    ['at_mlp_in',           'at_mlp_out'],
    ['at_mlp_out',          'action_time_tokens'],

    // ----- Suffix concat (state + action_time) -----
    // route state_token around the action/time mixing area instead of through it
    ['state_token',         'suffix_concat',
      { path: 'M 700 230 L 700 615 L 960 615 L 960 640' }],
    ['action_time_tokens',  'suffix_concat'],
    ['suffix_concat',       'suffix_tokens'],

    // ----- Into joint Gemma -----
    ['prefix_tokens', 'pre_norm_p'],
    ['suffix_tokens', 'pre_norm_s'],

    // ----- DualBlock: per-expert pre-norm + qkv -----
    ['pre_norm_p', 'qkv_p'],
    ['pre_norm_s', 'qkv_s'],

    // ----- DualBlock: joint attention -----
    ['qkv_p',       'attn_concat'],
    ['qkv_s',       'attn_concat'],
    ['attn_concat', 'rope'],
    ['rope',        'sdpa'],
    ['sdpa',        'split_attn'],

    // ----- DualBlock: split → per-expert out + residual -----
    ['split_attn', 'out_p'],
    ['split_attn', 'out_s'],

    // Prefix continued
    ['out_p',  'res1_p'],
    ['res1_p', 'ffn_p'],
    ['ffn_p',  'res2_p'],

    // Suffix continued
    ['out_s',  'res1_s'],
    ['res1_s', 'ffn_s'],
    ['ffn_s',  'res2_s'],

    // ----- After 18 layers: final norm per expert -----
    ['res2_p', 'final_norm_p'],
    ['res2_s', 'final_norm_s'],
    ['final_norm_p', 'prefix_out'],
    ['final_norm_s', 'suffix_out'],

    // ----- Output / sampler -----
    ['suffix_out',     'slice_action'],
    ['slice_action',   'action_out_proj'],
    ['action_out_proj','velocity'],
    ['velocity',       'euler'],
    ['euler',          'action_pred'],

    // ----- DualBlock × 18 loops (per expert) -----
    // prefix expert: from res2_p (bottom-left) up around the left side of dualblock back to pre_norm_p (top-left)
    ['res2_p', 'pre_norm_p',
      { loop: true, path: 'M 140 1382 L 80 1382 L 80 938 L 140 938' }],
    // suffix expert: from res2_s (bottom-right) up around the right side of dualblock back to pre_norm_s (top-right)
    ['res2_s', 'pre_norm_s',
      { loop: true, path: 'M 1360 1382 L 1420 1382 L 1420 938 L 1360 938' }],

    // ----- Flow-matching sampling loop: euler step → noisy_actions -----
    ['euler', 'noisy_actions',
      { loop: true, path: 'M 1360 1760 L 1480 1760 L 1480 55 L 1020 55' }],
  ],

  // DualBlock is a sub-module of the Joint Gemma transformer; hovering
  // any DualBlock node also lights up the outer Gemma rectangle.
  moduleParents: {
    dualblock: 'gemma',
  },
};
