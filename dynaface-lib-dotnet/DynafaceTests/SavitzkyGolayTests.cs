namespace DynafaceTests;

// Verifies SavitzkyGolay.Filter against scipy.signal.savgol_filter(..., mode="interp")
// reference vectors computed once via scipy 1.17.1 on a fixed synthetic signal (see
// the generating script referenced in the plan notes). This is the single highest-risk
// hand-rolled algorithm in the port — edge-mode handling is easy to get subtly wrong
// and there's no independent derivation to fall back on, only "does it match scipy."
public class SavitzkyGolayTests
{
    // x[i] = i*1.3 + 5*sin(i*0.5), i=0..24
    static readonly double[] Input =
    {
        0.0000000000, 3.6971276930, 6.8073549240, 8.8874749330, 9.7464871341,
        9.4923607205, 8.5056000403, 7.3460838616, 6.6159875235, 6.8123494117,
        8.2053786267, 10.7722983721, 14.2029225090, 17.9755999404, 21.4849329936,
        24.1899998839, 25.7467912331, 26.0924355631, 25.4605924262, 24.3242443977,
        23.2798944456, 22.9015212001, 23.6000489672, 25.5227391266, 28.5171354100,
    };

    static readonly double[] W9Deriv0 =
    {
        0.3173970068, 3.7904825002, 6.4684587794, 8.3513258444, 9.4390836952,
        9.2900370476, 8.4578920247, 7.4646720895, 6.8718374608, 7.1428200707,
        8.5295592644, 11.0108182623, 14.2973836639, 17.9028749748, 21.2628275154,
        23.8728930604, 25.4123218743, 25.8224934329, 25.3212687727, 24.3496505104,
        23.4638100219, 23.7721365195, 24.5561667695, 25.8159007719, 27.5513385267,
    };

    static readonly double[] W9Deriv1 =
    {
        3.8706401005, 3.0755308863, 2.2804216721, 1.4853124579, 0.6902032437,
        0.1260519239, -0.1506759639, -0.0722277817, 0.3421896196, 0.9911124068,
        1.7156616497, 2.3384424240, 2.7069762760, 2.7310332656, 2.4047234027,
        1.8079387223, 1.0867929277, 0.4178476603, -0.0351159483, -0.1611966089,
        0.0704746214, 0.5461783738, 1.0218821262, 1.4975858786, 1.9732896310,
    };

    static readonly double[] W9Deriv2 =
    {
        -0.7951092142, -0.7951092142, -0.7951092142, -0.7951092142, -0.7951092142,
        -0.5233169062, -0.1233983682, 0.3067323939, 0.6617643684, 0.8547733457,
        0.8385039966, 0.6169396254, 0.2443269173, -0.1881055413, -0.5744832030,
        -0.8202073408, -0.8651161159, -0.6982142938, -0.3603652615, 0.0657137550,
        0.4757037524, 0.4757037524, 0.4757037524, 0.4757037524, 0.4757037524,
    };

    static readonly double[] W13Deriv0 =
    {
        3.2366251537, 4.2805890549, 5.2461827270, 6.1334061699, 6.9422593836,
        7.6727423682, 8.3248551235, 7.7953630685, 7.5852916352, 8.0643589759,
        9.4335580368, 11.6759464742, 14.5607947319, 17.7000766827, 20.6434719580,
        22.9886200788, 24.4796323345, 25.0697422631, 24.9327557122, 25.4859812632,
        25.8103390542, 25.9058290851, 25.7724513560, 25.4102058668, 24.8190926177,
    };

    static readonly double[] W13Deriv1 =
    {
        1.0831490159, 1.0047787867, 0.9264085575, 0.8480383283, 0.7696680991,
        0.6912978699, 0.6129276407, 0.6500824424, 0.8463597287, 1.1537039748,
        1.4968665901, 1.7918293981, 1.9663752164, 1.9777691410, 1.8232215419,
        1.5405710613, 1.1990203948, 0.8821930573, 0.6676594310, 0.4387916710,
        0.2099239109, -0.0189438491, -0.2478116091, -0.4766793692, -0.7055471292,
    };

    static readonly double[] W13Deriv2 =
    {
        -0.0783702292, -0.0783702292, -0.0783702292, -0.0783702292, -0.0783702292,
        -0.0783702292, -0.0783702292, 0.1948055582, 0.4202861509, 0.5428660359,
        0.5325333821, 0.3918179837, 0.1551718777, -0.1194657157, -0.3648539355,
        -0.5209131871, -0.5494347230, -0.4434354765, -0.2288677600, -0.2288677600,
        -0.2288677600, -0.2288677600, -0.2288677600, -0.2288677600, -0.2288677600,
    };

    const double Tolerance = 1e-6;

    static void AssertClose(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.Math.Abs(expected[i] - actual[i]) < Tolerance,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    [Fact]
    public void Filter_Window9_Deriv0_MatchesScipy() => AssertClose(W9Deriv0, SavitzkyGolay.Filter(Input, 9, 2, 0));

    [Fact]
    public void Filter_Window9_Deriv1_MatchesScipy() => AssertClose(W9Deriv1, SavitzkyGolay.Filter(Input, 9, 2, 1));

    [Fact]
    public void Filter_Window9_Deriv2_MatchesScipy() => AssertClose(W9Deriv2, SavitzkyGolay.Filter(Input, 9, 2, 2));

    [Fact]
    public void Filter_Window13_Deriv0_MatchesScipy() => AssertClose(W13Deriv0, SavitzkyGolay.Filter(Input, 13, 2, 0));

    [Fact]
    public void Filter_Window13_Deriv1_MatchesScipy() => AssertClose(W13Deriv1, SavitzkyGolay.Filter(Input, 13, 2, 1));

    [Fact]
    public void Filter_Window13_Deriv2_MatchesScipy() => AssertClose(W13Deriv2, SavitzkyGolay.Filter(Input, 13, 2, 2));
}
