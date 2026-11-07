using System.Collections.Generic;

namespace DynafaceTests;

// Exercises the item-level enable/disable system (mirrors dynaface-lib's
// measures_base.py MeasureBase/MeasureItem contract) using MeasureDentalArea as a
// representative multi-item measure.
public class FaceMeasureBaseTests
{
    [Fact]
    public void SetEnabled_False_CascadesToAllItems()
    {
        var measure = new MeasureDentalArea();
        measure.SetEnabled(false);

        Assert.False(measure.Enabled);
        foreach (var item in measure.Items)
            Assert.False(item.Enabled);
    }

    [Fact]
    public void SetItemEnabled_TogglesOnlyNamedItem()
    {
        var measure = new MeasureDentalArea();
        measure.SetItemEnabled("dental_ratio", false);

        Assert.False(measure.IsEnabled("dental_ratio"));
        Assert.True(measure.IsEnabled("dental_area"));
        Assert.True(measure.IsEnabled("dental_left"));
    }

    [Fact]
    public void IsEnabled_UnknownItemName_DefaultsToTrue()
    {
        var measure = new MeasureDentalArea();
        Assert.True(measure.IsEnabled("not_a_real_item"));
    }

    [Fact]
    public void UpdateForType_SetsEveryItemToTheMeasuresOwnFlag()
    {
        // Matches dynaface-lib's update_for_type: it syncs the items' flags from the
        // measure's first, so an item without views of its own gets the MEASURE's
        // IsLateral/IsFrontal/IsQuarter flag, whatever the item held before.
        var measure = new MeasurePosition(); // IsFrontal, IsLateral and IsQuarter
        foreach (var item in measure.Items) item.IsLateral = item.IsFrontal = item.IsQuarter = false;
        measure.UpdateForType(lateral: true);
        foreach (var item in measure.Items) Assert.True(item.Enabled);

        measure.UpdateForType(lateral: false);
        foreach (var item in measure.Items) Assert.True(item.Enabled);

        measure.UpdateForType(lateral: false, quarter: true);
        foreach (var item in measure.Items) Assert.True(item.Enabled);

        var fai = new MeasureFAI(); // frontal only
        fai.UpdateForType(lateral: false, quarter: true);
        Assert.DoesNotContain(fai.Items, item => item.Enabled);
        fai.UpdateForType(lateral: false);
        Assert.All(fai.Items, item => Assert.True(item.Enabled));
        fai.UpdateForType(lateral: true, quarter: true); // lateral first
        Assert.DoesNotContain(fai.Items, item => item.Enabled);
    }

    // FaceMeasureBase.Suits reads the measure's flags with one rule: lateral, else
    // quarter, else frontal (dynaface-lib's MeasureBase.suits).
    [Fact]
    public void Suits_LateralElseQuarterElseFrontal()
    {
        var position = new MeasurePosition();
        Assert.True(position.Suits(false) && position.Suits(true) && position.Suits(false, true));

        var fai = new MeasureFAI();
        Assert.True(fai.Suits(false));
        Assert.False(fai.Suits(false, quarter: true));
        Assert.False(fai.Suits(true));

        var lateral = new MeasureLateral();
        Assert.False(lateral.Suits(false));
        Assert.False(lateral.Suits(false, quarter: true));
        Assert.True(lateral.Suits(true));
        Assert.True(lateral.Suits(true, quarter: true)); // quarter is ignored for a profile

        var landmarks = new MeasureLandmarks();
        Assert.True(landmarks.Suits(false) && landmarks.Suits(false, quarter: true));
        Assert.False(landmarks.Suits(true));
    }

    // A quarter (three-quarter) face has no millimeter, so only the measures without one
    // suit it, as in dynaface-lib.
    [Fact]
    public void TheQuarterMeasuresAreTheOnesWithoutMillimeters()
    {
        var measures = DynafaceMeasures.AllMeasures();
        Assert.Equal(new[] { "POSITION", "POSE", "LANDMARKS", "CHEEK MEASURES" },
            measures.Where(m => m.IsQuarter).Select(m => m.Label));
        foreach (var measure in measures)
        {
            Assert.Equal(measure.IsQuarter, measure.Suits(false, quarter: true));
            // Each measure's items follow its view, narrowed to an item's own views (the
            // Cheek Measures' frontal items don't suit a quarter face).
            Assert.All(measure.Items, item => Assert.Equal(
                measure.IsQuarter && (item.Views == null || item.Views.Contains(FacePose.Quarter)), item.IsQuarter));
        }
    }

    [Fact]
    public void Outputs_AreTheItemsThatArentDisplayOnly()
    {
        Assert.Equal(new[] { "tilt", "px2mm", "pd" }, new MeasurePosition().Outputs());
        var lateral = new MeasureLateral();
        Assert.DoesNotContain("tip_proj", lateral.Outputs());
        Assert.Equal(lateral.Items.Where(i => !i.DisplayOnly).Select(i => i.Name), lateral.Outputs());
    }

    // dynaface-lib's AnalyzeFace.quarter: a Quarter pose with landmarks.
    [Fact]
    public void ContextIsQuarter_ForAQuarterPoseWithLandmarks()
    {
        var photo = new FaceImage(new Rgba32[64 * 64], 64, 64);
        Assert.True(new FaceMeasureContext(photo, TestHelpers.MakeLandmarks(), pose: FacePose.Quarter).IsQuarter);
        Assert.False(new FaceMeasureContext(photo, new Vec2[0], pose: FacePose.Quarter).IsQuarter);
        Assert.False(new FaceMeasureContext(photo, null!, pose: FacePose.Quarter).IsQuarter);
        Assert.False(new FaceMeasureContext(photo, TestHelpers.MakeLandmarks()).IsQuarter);
        Assert.False(new FaceMeasureContext(photo, TestHelpers.MakeLandmarks(), pose: FacePose.Frontal).IsQuarter);
    }

    // On a quarter face MeasurePosition reports no scale (its pupil distance is
    // foreshortened by the turn) and doesn't draw one; tilt and the pixel pd stay, as in
    // dynaface-lib's AnalyzePosition.
    [Fact]
    public void Position_ReportsNoScaleOnAQuarterFace()
    {
        var landmarks = TestHelpers.MakeLandmarks();
        landmarks[DynafaceConstants.LmRightPupil] = new Vec2(380f, 480f);
        landmarks[DynafaceConstants.LmLeftPupil] = new Vec2(620f, 490f);
        var photo = new FaceImage(new Rgba32[1024 * 1024], 1024, 1024);
        var frontal = new FaceMeasureContext(photo, landmarks, pose: FacePose.Frontal, pupilDistMm: 63f);
        var quarter = new FaceMeasureContext(photo, landmarks, pose: FacePose.Quarter, pupilDistMm: 63f);

        var measure = new MeasurePosition();
        var onFrontal = measure.Calc(frontal, render: true);
        var onQuarter = measure.Calc(quarter, render: true);

        Assert.Equal(63.0 / onFrontal["pd"], onFrontal["px2mm"], 5);
        Assert.True(double.IsNaN(onQuarter["px2mm"]));
        Assert.Equal(onFrontal["pd"], onQuarter["pd"]);
        Assert.Equal(onFrontal["tilt"], onQuarter["tilt"]);
        Assert.Contains(frontal.TextLines, line => line.Contains("px2mm"));
        Assert.DoesNotContain(quarter.TextLines, line => line.Contains("px2mm"));
        Assert.Contains(quarter.TextLines, line => line.Contains("pd:"));
    }

    [Fact]
    public void DisablingAnItem_SuppressesItsResultLine_ButStillReturnsAllValues()
    {
        // Matches dynaface-lib's filter_measurements: the returned dict always
        // includes every declared item regardless of its enabled state — only
        // rendering (pixels/sidebar text) is gated by IsEnabled().
        var measure = new MeasureFAI();
        measure.SetItemEnabled("fai", false);
        var ctx = TestHelpers.BuildContext();

        Dictionary<string, double> result = measure.Calc(ctx, render: true);

        Assert.True(result.ContainsKey("fai"));
        Assert.Empty(ctx.TextLines); // suppressed since the "fai" item is disabled
    }

    [Fact]
    public void FaceMeasureBase_DefaultsEnabledTrue()
    {
        // dynaface-lib's MeasureBase.__init__ defaults self.enabled=True.
        var measure = new MeasureFAI();
        Assert.True(measure.Enabled);
    }
}
