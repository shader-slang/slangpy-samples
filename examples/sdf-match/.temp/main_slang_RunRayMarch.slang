/*
Name                 | Index      | PyType                         | SlType                         | VType                          | Shape                | Call Dim   | VMap                
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
screenSize           | 0          | vector<float,2>                |                                | vector<float,2>                | [2]                  | 0          | []                  
gridId               | 1          | CallIdArg                      |                                | vector<int,2>                  | []                   | 2          | [0, 1]              
params               | 2          | int                            |                                | Ptr<TinyMLP_Params>            | []                   | 0          | []                  
_result              | -1         | RWTexture2D<vector<float,4>>   |                                | vector<float,4>                | [-1, -1, 4]          | 2          | [0, 1]              
*/

import "slangpy";
import "main.slang";

export static const int call_data_len = 2;
export static const int call_group_size = 1;
export static const int[call_data_len] call_group_strides = {
};
export static const int[call_data_len] call_group_shape_vector = {
};


typealias _t_screenSize = VectorValueType<float,2>;
static const int _m_screenSize = 0;
typealias _t_gridId = CallIdArg;
static const int[] _m_gridId = { 0,1 };
typealias _t_params = ValueType<Ptr<TinyMLP_Params>>;
static const int _m_params = 0;
typealias _t__result = RWTexture2DType<vector<float,4>>;
static const int[] _m__result = { 0,1 };

struct CallData
{
    int[2] _grid_stride;
    int[2] _grid_dim;
    int[2] _call_dim;
    uint3 _thread_count;
    _t_screenSize screenSize;
    _t_gridId gridId;
    _t_params params;
    _t__result _result;
}
ParameterBlock<CallData> call_data;




void _trampoline(Context __slangpy_context__)
{
    vector<float,4> _result;
    vector<float,2> screenSize;
    vector<int,2> gridId;
    Ptr<TinyMLP_Params> params;
    call_data.screenSize.load(__slangpy_context__.map(_m_screenSize), screenSize);
    call_data.gridId.load(__slangpy_context__.map(_m_gridId), gridId);
    call_data.params.load(__slangpy_context__.map(_m_params), params);
    _result = RunRayMarch(screenSize, gridId, params);
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

