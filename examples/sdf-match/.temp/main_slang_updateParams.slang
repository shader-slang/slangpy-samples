/*
Name                 | Index      | PyType                         | SlType                         | VType                          | Shape                | Call Dim   | VMap                
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params               | 0          | int                            |                                | Ptr<TinyMLP_Params>            | []                   | 0          | []                  
state                | 1          | RWNDBuffer<AdamState, 1>       |                                | AdamState                      | [-1]                 | 1          | [0]                 
batch_size           | 2          | int                            |                                | int                            | []                   | 0          | []                  
beta1                | 3          | float                          |                                | float                          | []                   | 0          | []                  
beta2                | 4          | float                          |                                | float                          | []                   | 0          | []                  
*/

import "slangpy";
import "main.slang";

export static const int call_data_len = 1;
export static const int call_group_size = 1;
export static const int[call_data_len] call_group_strides = {
};
export static const int[call_data_len] call_group_shape_vector = {
};


typealias _t_params = ValueType<Ptr<TinyMLP_Params>>;
static const int _m_params = 0;
typealias _t_state = RWNDBuffer<AdamState,1>;
static const int[] _m_state = { 0 };
typealias _t_batch_size = ValueType<int>;
static const int _m_batch_size = 0;
typealias _t_beta1 = ValueType<float>;
static const int _m_beta1 = 0;
typealias _t_beta2 = ValueType<float>;
static const int _m_beta2 = 0;

struct CallData
{
    int[1] _grid_stride;
    int[1] _grid_dim;
    int[1] _call_dim;
    uint3 _thread_count;
    _t_params params;
    _t_state state;
    _t_batch_size batch_size;
    _t_beta1 beta1;
    _t_beta2 beta2;
}
ParameterBlock<CallData> call_data;




void _trampoline(Context __slangpy_context__)
{
    Ptr<TinyMLP_Params> params;
    AdamState state;
    int batch_size;
    float beta1;
    float beta2;
    call_data.params.load(__slangpy_context__.map(_m_params), params);
    call_data.state.load(__slangpy_context__.map(_m_state), state);
    call_data.batch_size.load(__slangpy_context__.map(_m_batch_size), batch_size);
    call_data.beta1.load(__slangpy_context__.map(_m_beta1), beta1);
    call_data.beta2.load(__slangpy_context__.map(_m_beta2), beta2);
    updateParams(params, state, batch_size, beta1, beta2);
    call_data.state.store(__slangpy_context__.map(_m_state), state);
}


[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID, int3 flat_call_group_id: SV_GroupID, int flat_call_group_thread_id: SV_GroupIndex)
{
    if (any(flat_call_thread_id >= call_data._thread_count)) return;
    
        if (!init_thread_local_call_shape_info(flat_call_group_thread_id,
            flat_call_group_id, flat_call_thread_id, call_data._grid_stride,
            call_data._grid_dim, call_data._call_dim))
            return;
    Context __slangpy_context__ = {flat_call_thread_id, CallShapeInfo::get_call_id().shape};
    _trampoline(__slangpy_context__);
}

