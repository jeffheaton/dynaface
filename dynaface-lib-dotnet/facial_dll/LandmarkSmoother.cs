using System;
using System.Collections.Generic;

// A moving average of the last few frames' landmarks, which steadies their jitter from
// frame to frame, as the Dynaface apps' data-smoothing setting does. Give it the
// landmarks of the cropped face (the 1024x1024 crop), where the head moving within the
// frame doesn't blur them. Mirrors dynaface-lib's dynaface.video.LandmarkSmoother.
public sealed class LandmarkSmoother
{
    readonly Queue<Vec2[]> _window = new Queue<Vec2[]>();

    // size: how many frames to average, this one included; 1 turns smoothing off.
    public LandmarkSmoother(int size)
    {
        if (size < 1)
            throw new ArgumentOutOfRangeException(nameof(size), size, "The size must be at least 1.");
        Size = size;
    }

    public int Size { get; }

    // Adds a frame's landmarks and returns the average of the last Size frames' landmarks,
    // this frame's included, truncated toward zero to whole pixels like dynaface-lib's.
    // With a Size of 1, a copy of the landmarks as they are.
    public Vec2[] Add(Vec2[] landmarks)
    {
        if (Size == 1) return (Vec2[])landmarks.Clone();

        _window.Enqueue((Vec2[])landmarks.Clone());
        while (_window.Count > Size) _window.Dequeue();

        int n = _window.Count;
        var average = new Vec2[landmarks.Length];
        for (int i = 0; i < landmarks.Length; i++)
        {
            double sumX = 0, sumY = 0;
            foreach (Vec2[] frame in _window)
            {
                sumX += frame[i].X;
                sumY += frame[i].Y;
            }
            average[i] = new Vec2((int)(sumX / n), (int)(sumY / n));
        }
        return average;
    }

    // Forgets the earlier frames, so the next one starts a new average.
    public void Reset() => _window.Clear();
}
