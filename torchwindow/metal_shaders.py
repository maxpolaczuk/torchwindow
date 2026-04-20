"""Metal Shading Language source for the Metal-backed Window.

Mirrors torchwindow/shaders.py (GLSL): a single fullscreen triangle
that samples the tensor-backed MTLTexture. Kept in Python as a string
so the Metal backend can compile it at runtime via MTLDevice
newLibraryWithSource, matching how create_shader_program() works for
the GL path.
"""

from __future__ import annotations

MSL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 texcoords;
};

// Single fullscreen triangle — covers the NDC quad without a vertex buffer.
// Positions and UVs match torchwindow/shaders.py (GLSL).
vertex Varyings vs_main(uint vid [[vertex_id]]) {
    const float4 positions[3] = {
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 3.0,  1.0, 0.0, 1.0),
        float4(-1.0, -3.0, 0.0, 1.0),
    };
    const float2 uvs[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0),
    };
    Varyings out;
    out.position = positions[vid];
    out.texcoords = uvs[vid];
    return out;
}

fragment float4 fs_main(
    Varyings in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]]
) {
    return tex.sample(samp, in.texcoords);
}
"""
