using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Implements IDynafaceInference using Microsoft.ML.OnnxRuntime.
// Takes separate ONNX model paths for BlazeFace, SPIGA, and U^2-Net at construction
// time — 3 independent InferenceSessions, one per network, never fused.
// Tensor names are auto-detected from model metadata; pass explicit overrides if needed.
//
// SPIGA requires three inputs per inference:
//   image      [1, 3, 256, 256]  — NCHW float32 in [0, 1]
//   model3d    [1, 98, 3]        — fixed WFLW 3D face template  (baked in as constant)
//   cam_matrix [1, 3, 3]         — fixed camera intrinsics      (baked in as constant)
// and two outputs: "landmarks" and "pose".
//
// U^2-Net has 7 outputs (d0..d6, multi-scale side outputs from training); the first
// declared output (auto-detected, not hardcoded) is the final merged saliency map.
public class OnnxDynafaceInference : IDynafaceInference
{
    readonly InferenceSession _blazeFaceSession;
    readonly InferenceSession _spigaSession;
    readonly InferenceSession _u2netSession;
    readonly string _blazeInputName;
    readonly string _blazeRegressorsName;
    readonly string _blazeScoresName;
    readonly string _u2netInputName;
    readonly string _u2netOutputName;

    public bool IsReady => _blazeFaceSession != null && _spigaSession != null && _u2netSession != null;

    public OnnxDynafaceInference(string blazeFacePath, string spigaPath, string u2netPath,
        string blazeInputName      = null,
        string blazeRegressorsName = null,
        string blazeScoresName     = null)
    {
        _blazeFaceSession   = new InferenceSession(blazeFacePath);
        _spigaSession       = new InferenceSession(spigaPath);
        _u2netSession       = new InferenceSession(u2netPath);
        _blazeInputName      = blazeInputName      ?? _blazeFaceSession.InputMetadata.Keys.First();
        string[] outs        = _blazeFaceSession.OutputMetadata.Keys.ToArray();
        _blazeRegressorsName = blazeRegressorsName ?? outs[0];
        _blazeScoresName     = blazeScoresName     ?? outs[1];
        _u2netInputName       = _u2netSession.InputMetadata.Keys.First();
        _u2netOutputName      = _u2netSession.OutputMetadata.Keys.First();
    }

    public string DescribeTensors()
    {
        var spigaIns  = string.Join(", ", _spigaSession.InputMetadata.Keys);
        var spigaOuts = string.Join(", ", _spigaSession.OutputMetadata.Keys);
        return $"BlazeFace: input={_blazeInputName}  regressors={_blazeRegressorsName}  scores={_blazeScoresName}\n" +
               $"SPIGA:     inputs=[{spigaIns}]  outputs=[{spigaOuts}]\n" +
               $"U2Net:     input={_u2netInputName}  output={_u2netOutputName}";
    }

    public (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor)
    {
        try
        {
            var t      = new DenseTensor<float>(tensor, new[] { 1, 128, 128, 3 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_blazeInputName, t) };
            using var results = _blazeFaceSession.Run(inputs);
            float[] regs   = results.First(r => r.Name == _blazeRegressorsName).AsTensor<float>().ToArray();
            float[] scores = results.First(r => r.Name == _blazeScoresName).AsTensor<float>().ToArray();
            return (regs, scores);
        }
        catch { return null; }
    }

    public (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor)
    {
        try
        {
            var imgT = new DenseTensor<float>(imageTensor, new[] { 1, 3, 256, 256 });
            var m3dT = new DenseTensor<float>(_model3d,    new[] { 1, 98, 3 });
            var camT = new DenseTensor<float>(_camMatrix,  new[] { 1, 3, 3 });

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("image",      imgT),
                NamedOnnxValue.CreateFromTensor("model3d",    m3dT),
                NamedOnnxValue.CreateFromTensor("cam_matrix", camT),
            };

            using var results = _spigaSession.Run(inputs);
            float[] landmarks = results.First(r => r.Name == "landmarks").AsTensor<float>().ToArray();
            float[] pose      = results.First(r => r.Name == "pose").AsTensor<float>().ToArray();
            return (landmarks, pose);
        }
        catch { return null; }
    }

    public float[] RunU2Net(float[] imageTensor)
    {
        try
        {
            var t      = new DenseTensor<float>(imageTensor, new[] { 1, 3, 320, 320 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_u2netInputName, t) };
            using var results = _u2netSession.Run(inputs);
            return results.First(r => r.Name == _u2netOutputName).AsTensor<float>().ToArray();
        }
        catch { return null; }
    }

    public void Dispose()
    {
        _blazeFaceSession?.Dispose();
        _spigaSession?.Dispose();
        _u2netSession?.Dispose();
    }

    // -----------------------------------------------------------------------
    // WFLW 3D face template and camera intrinsics
    // Source: SPIGA repo, dataset=wflw, ftmap_size=(64,64), focal_ratio=1.5
    // -----------------------------------------------------------------------

    static readonly float[] _camMatrix = new float[]
    {
        96f, 0f, 32f,
        0f, 96f, 32f,
        0f,  0f,  1f,
    };

    // 98 rows × 3 values (x, y, z) in row-major order
    static readonly float[] _model3d = new float[]
    {
        -0.853184236268f,  0.710460614932f, -0.393345437621f,
        -0.865903665506f,  0.697935135807f, -0.281548075586f,
        -0.878623094748f,  0.685409656727f, -0.169750713576f,
        -0.900489075720f,  0.646541048786f, -0.054976885980f,
        -0.922355056610f,  0.607672440835f,  0.059796941671f,
        -0.921595827864f,  0.576668701754f,  0.161148197750f,
        -0.920836599128f,  0.545664962608f,  0.262499453743f,
        -0.891987904048f,  0.495939744597f,  0.350772314723f,
        -0.863139209001f,  0.446214526602f,  0.439045175772f,
        -0.796026245014f,  0.390325238349f,  0.502547532515f,
        -0.728913281017f,  0.334435950088f,  0.566049889293f,
        -0.657186170535f,  0.285302127936f,  0.609874056230f,
        -0.585459059997f,  0.236168305748f,  0.653698223146f,
        -0.536840395466f,  0.182237836481f,  0.682871772143f,
        -0.488221730878f,  0.128307367184f,  0.712045321068f,
        -0.474793009911f,  0.065927929287f,  0.725020759991f,
        -0.461364289002f,  0.003548491464f,  0.737996198981f,
        -0.463813244263f, -0.063825309161f,  0.735566080866f,
        -0.466262199443f, -0.131199109787f,  0.733135962680f,
        -0.513521912927f, -0.194286440465f,  0.699811948342f,
        -0.560781626342f, -0.257373771199f,  0.666487934013f,
        -0.613723257723f, -0.308598755049f,  0.618041459290f,
        -0.666664889106f, -0.359823738861f,  0.569594984479f,
        -0.714432235344f, -0.407851339087f,  0.522433885778f,
        -0.762199581605f, -0.455878939366f,  0.475272786996f,
        -0.828822422342f, -0.519515088045f,  0.365356986341f,
        -0.895445263130f, -0.583151236655f,  0.255441185628f,
        -0.910235122162f, -0.609093253652f,  0.166017120718f,
        -0.925024981150f, -0.635035270609f,  0.076593055883f,
        -0.915635837119f, -0.661799095895f, -0.025251322986f,
        -0.906246693121f, -0.688562921268f, -0.127095701883f,
        -0.904118559673f, -0.700222989101f, -0.229632887961f,
        -0.901990426154f, -0.711883056935f, -0.332170073985f,
        -0.466456539275f,  0.552168877879f, -0.483792334207f,
        -0.308160933540f,  0.459810924980f, -0.561780416182f,
        -0.216753374882f,  0.360178576451f, -0.566290707559f,
        -0.148694799479f,  0.249770054109f, -0.530336745190f,
        -0.124216132105f,  0.101941089981f, -0.482471777751f,
        -0.132953775070f,  0.100790757462f, -0.429593539303f,
        -0.157432442443f,  0.248619721590f, -0.477458506742f,
        -0.225491017847f,  0.359028243932f, -0.513412469111f,
        -0.316898576504f,  0.458660592461f, -0.508902177734f,
        -0.131673602172f, -0.097976488432f, -0.465935806417f,
        -0.135680573937f, -0.214281256932f, -0.497130488240f,
        -0.223000042464f, -0.373011222167f, -0.520379035114f,
        -0.319251580460f, -0.501607216528f, -0.500371303275f,
        -0.465084060803f, -0.573625223791f, -0.457842172777f,
        -0.310513937496f, -0.500456884008f, -0.447493064827f,
        -0.231737685429f, -0.374161554686f, -0.467500796666f,
        -0.144418216902f, -0.215431589451f, -0.444252249792f,
        -0.140411245136f, -0.099126820952f, -0.413057567969f,
        -0.139142361862f, -0.008435261941f, -0.416310824526f,
        -0.106471830333f, -0.007312178351f, -0.267155736618f,
        -0.054453930590f, -0.001901740064f, -0.142359799356f,
         0.000000000000f,  0.000000000000f,  0.000000000000f,
        -0.235537796980f,  0.118903311783f,  0.079278454225f,
        -0.171210015652f,  0.050430024294f,  0.104665185108f,
        -0.154139340027f, -0.003126570926f,  0.111845126622f,
        -0.173275169304f, -0.068754398342f,  0.099545856333f,
        -0.238513462350f, -0.148988810614f,  0.073784851484f,
        -0.383328256738f,  0.406302776084f, -0.360389456414f,
        -0.310253684249f,  0.344774145080f, -0.384294131116f,
        -0.298449428843f,  0.303148027928f, -0.398970401478f,
        -0.286645173373f,  0.261521910707f, -0.396020592392f,
        -0.330243237010f,  0.160787895283f, -0.371046071665f,
        -0.323105980144f,  0.245621576158f, -0.334421719595f,
        -0.331524487470f,  0.286856859665f, -0.324384347457f,
        -0.339942994736f,  0.328092143155f, -0.331973054752f,
        -0.314140711048f, -0.163880511996f, -0.340747336431f,
        -0.279030578852f, -0.250844605212f, -0.392490504075f,
        -0.282436250152f, -0.295807479997f, -0.394548614690f,
        -0.285841921386f, -0.340770354713f, -0.378980645804f,
        -0.377158168879f, -0.432352000574f, -0.348665008377f,
        -0.318530865652f, -0.333467711265f, -0.328033806608f,
        -0.315834032602f, -0.290044551442f, -0.322730607606f,
        -0.313137199533f, -0.246621391603f, -0.335053488012f,
        -0.354306324759f,  0.222983201100f,  0.277627584818f,
        -0.270218764688f,  0.157062857603f,  0.248737641675f,
        -0.225237980339f,  0.081635485618f,  0.236184984280f,
        -0.212609820801f,  0.003915777595f,  0.244998123230f,
        -0.208501270490f, -0.062942106644f,  0.230774971513f,
        -0.235471195707f, -0.135507933036f,  0.240646328002f,
        -0.332281891846f, -0.240381217797f,  0.283709533359f,
        -0.283978333058f, -0.143706630542f,  0.329461104519f,
        -0.256464077312f, -0.084139951946f,  0.355354733250f,
        -0.250423014186f, -0.008137040189f,  0.360357397838f,
        -0.269661366141f,  0.077826069065f,  0.349041413147f,
        -0.322383820174f,  0.157111645515f,  0.311971743235f,
        -0.341170846657f,  0.182660017758f,  0.278377925956f,
        -0.270854173882f,  0.075943475389f,  0.283519721316f,
        -0.254428375697f,  0.002670132255f,  0.286917503124f,
        -0.257845545665f, -0.068506200339f,  0.285630836758f,
        -0.314605542850f, -0.194170296105f,  0.279641061938f,
        -0.258696545260f, -0.068986247483f,  0.288076969827f,
        -0.256229467080f,  0.002739134144f,  0.289157237470f,
        -0.273027820052f,  0.076698969737f,  0.286311968999f,
        -0.328919887677f,  0.291183407723f, -0.363024170951f,
        -0.314639907590f, -0.294656095888f, -0.353995131537f,
    };
}
