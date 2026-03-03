// SPDX-License-Identifier: Apache-2.0
import neural;

export struct InputVec : IVector<float> = InlineVector<float, 4>;
export struct HiddenVec : IVector<float> = InlineVector<float, 32>;
export struct OutputVec : IVector<float> = InlineVector<float, 3>;

typealias ConcreteInput = InlineVector<float, 4>;
typealias ConcreteOutput = InlineVector<float, 3>;

// WAR: InlineVector.__init is not yet differentiable, operator[] is internal
InputVec _makeInputVec(float data[4]) { return bit_cast<InputVec>(ConcreteInput(data)); }
float3 _extractFloat3(OutputVec v) { ConcreteOutput cv = bit_cast<ConcreteOutput>(v); return float3(cv[0], cv[1], cv[2]); }

[Differentiable] export InputVec arrayToInputVec(float data[4]) { return no_diff _makeInputVec(data); }

[BackwardDerivativeOf(arrayToInputVec)]
void arrayToInputVec_bwd(inout DifferentialPair<float[4]> dpData, InputVec.Differential dResult)
{
    ConcreteInput.Differential dr = bit_cast<ConcreteInput.Differential>(dResult);
    float dd[4];
    [ForceUnroll] for (int i = 0; i < 4; ++i) dd[i] = dpData.d[i] + dr[i];
    dpData = diffPair(dpData.p, dd);
}

[Differentiable] export float3 outputToFloat3(OutputVec v) { return no_diff _extractFloat3(v); }

[BackwardDerivativeOf(outputToFloat3)]
void outputToFloat3_bwd(inout DifferentialPair<OutputVec> dpV, float3 dResult)
{
    var concreteDp = bit_cast<DifferentialPair<ConcreteOutput>>(dpV);
    ConcreteOutput.Differential dv = concreteDp.d;
    dv[0] = dv[0] + dResult.x;
    dv[1] = dv[1] + dResult.y;
    dv[2] = dv[2] + dResult.z;
    dpV = bit_cast<DifferentialPair<OutputVec>>(diffPair(concreteDp.p, dv));
}
