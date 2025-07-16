/*
Name                 | Index      | PyType                         | SlType                         | VType                          | Shape                | Call Dim   | VMap                
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
state                | 0          | RWNDBuffer<AdamState, 1>       |                                | AdamState                      | [-1]                 | 1          | [0]                 
*/

import "slangpy";
import "example.slang";

export static const int call_data_len = 1;
export static const int call_group_size = 1;
export static const int[call_data_len] call_group_strides = {
};
export static const int[call_data_len] call_group_shape_vector = {
};


typealias _t_state = RWNDBuffer<AdamState,1>;
static const int[] _m_state = { 0 };

struct CallData
{
    int[1] _grid_stride;
    int[1] _grid_dim;
    int[1] _call_dim;
    uint3 _thread_count;
    _t_state state;
}
ParameterBlock<CallData> call_data;




void _trampoline(Context __slangpy_context__)
{
    AdamState state;
    call_data.state.load(__slangpy_context__.map(_m_state), state);
    clearAdamState(state);
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

