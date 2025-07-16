/*
Name                 | Index      | PyType                         | SlType                         | VType                          | Shape                | Call Dim   | VMap                
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params               | 0          | int                            |                                | Ptr<TinyMLP_Params>            | []                   | 0          | []                  
input                | 1          | RWTensor<float, 2>             |                                | float[3]                       | [-1, -1]             | 1          | [0]                 
_result              | -1         | RWNDBuffer<float, 1>           |                                | float                          | [-1]                 | 1          | [0]                 
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
typealias _t_input = Tensor<float, 2>;
static const int[] _m_input = { 0 };
typealias _t__result = RWNDBuffer<float,1>;
static const int[] _m__result = { 0 };

struct CallData
{
    int[1] _grid_stride;
    int[1] _grid_dim;
    int[1] _call_dim;
    uint3 _thread_count;
    _t_params params;
    _t_input input;
    _t__result _result;
}
ParameterBlock<CallData> call_data;




void _trampoline(Context __slangpy_context__)
{
    float _result;
    Ptr<TinyMLP_Params> params;
    float[3] input;
    call_data.params.load(__slangpy_context__.map(_m_params), params);
    call_data.input.load(__slangpy_context__.map(_m_input), input);
    _result = trainMLP(params, input);
    call_data._result.store(__slangpy_context__.map(_m__result), _result);
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

