[shader("compute")]
[numthreads(32, 1, 1)]
void main(
    uint tid: SV_DispatchThreadID,
    uniform uint N,
    StructuredBuffer<float> a,
    StructuredBuffer<float> b,
    RWStructuredBuffer<float> c
)
{
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}
