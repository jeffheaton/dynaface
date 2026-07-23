using System;

// General Savitzky-Golay filter, matching scipy.signal.savgol_filter(x, window_length=w,
// polyorder=p, deriv=d, mode="interp") — including scipy's "interp" edge handling,
// which is NOT simple padding/reflection: for the first/last (w-1)/2 samples, it
// refits the same-order polynomial to the edge's own w-sample window and evaluates
// that fit (or its derivative) at the actual sample offset, rather than reusing the
// sliding window's centered coefficients.
//
// Works uniformly for interior AND edge samples via one code path (fit a local
// polynomial, then evaluate its d-th derivative at whatever offset the query point
// sits at relative to that window's own center) rather than a separate "fast"
// dot-product shortcut for the interior — trading a little performance for a lot
// less risk of the two paths silently disagreeing.
public static class SavitzkyGolay
{
    public static double[] Filter(double[] x, int windowLength, int polyOrder, int deriv)
    {
        int n = x.Length;
        var result = new double[n];
        if (n == 0) return result;

        int w = windowLength;
        if (w > n) w = (n % 2 == 1) ? n : n - 1;
        if (w < 1) w = 1;
        int halfWin = (w - 1) / 2;

        double[,] pinv = ComputePseudoInverse(w, polyOrder);

        for (int i = 0; i < n; i++)
        {
            int startIdx;
            double t0;
            if (i < halfWin)
            {
                startIdx = 0;
                t0 = i - halfWin;
            }
            else if (i >= n - halfWin)
            {
                startIdx = n - w;
                t0 = i - (startIdx + halfWin);
            }
            else
            {
                startIdx = i - halfWin;
                t0 = 0;
            }

            double[] coeffs = PolyFit(x, startIdx, w, pinv, polyOrder);
            result[i] = EvaluateDerivative(coeffs, polyOrder, deriv, t0);
        }

        return result;
    }

    // H = (A^T A)^-1 A^T, the [polyOrder+1, w] pseudo-inverse of the window's
    // Vandermonde design matrix A[k,j] = (k-halfWin)^j, k=0..w-1, j=0..polyOrder.
    // Offsets are always relative to the window's OWN center (index halfWin within
    // whatever w-length slice is being fit) — the same H is reused for the sliding
    // interior window and both fixed edge windows.
    static double[,] ComputePseudoInverse(int w, int polyOrder)
    {
        int halfWin = (w - 1) / 2;
        int m = polyOrder + 1;

        var AtA = new double[m, m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
            {
                double sum = 0;
                for (int k = 0; k < w; k++)
                {
                    double t = k - halfWin;
                    sum += Math.Pow(t, i) * Math.Pow(t, j);
                }
                AtA[i, j] = sum;
            }

        double[,] AtAInv = Invert(AtA, m);

        var H = new double[m, w];
        for (int i = 0; i < m; i++)
            for (int k = 0; k < w; k++)
            {
                double t = k - halfWin;
                double sum = 0;
                for (int j = 0; j < m; j++)
                    sum += AtAInv[i, j] * Math.Pow(t, j);
                H[i, k] = sum;
            }
        return H;
    }

    // c[j] = coefficient of t^j in the fitted polynomial f(t), t relative to the
    // window's own center (absolute index startIdx+halfWin).
    static double[] PolyFit(double[] x, int startIdx, int w, double[,] pinv, int polyOrder)
    {
        int m = polyOrder + 1;
        var c = new double[m];
        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            for (int k = 0; k < w; k++) sum += pinv[i, k] * x[startIdx + k];
            c[i] = sum;
        }
        return c;
    }

    // f^(deriv)(t0) for f(t) = sum_j c[j] t^j.
    static double EvaluateDerivative(double[] c, int polyOrder, int deriv, double t0)
    {
        double result = 0;
        for (int j = deriv; j <= polyOrder; j++)
        {
            double fallingFactorial = 1;
            for (int k = 0; k < deriv; k++) fallingFactorial *= (j - k);
            result += c[j] * fallingFactorial * Math.Pow(t0, j - deriv);
        }
        return result;
    }

    // Gauss-Jordan inverse with partial pivoting. Only ever called on small
    // (polyOrder+1)-square matrices (3x3 for dynaface-lib's fixed polyorder=2).
    static double[,] Invert(double[,] m, int size)
    {
        var a = (double[,])m.Clone();
        var inv = new double[size, size];
        for (int i = 0; i < size; i++) inv[i, i] = 1.0;

        for (int col = 0; col < size; col++)
        {
            int pivotRow = col;
            double maxAbs = Math.Abs(a[col, col]);
            for (int r = col + 1; r < size; r++)
            {
                double v = Math.Abs(a[r, col]);
                if (v > maxAbs) { maxAbs = v; pivotRow = r; }
            }
            if (pivotRow != col)
            {
                for (int c = 0; c < size; c++)
                {
                    (a[col, c], a[pivotRow, c]) = (a[pivotRow, c], a[col, c]);
                    (inv[col, c], inv[pivotRow, c]) = (inv[pivotRow, c], inv[col, c]);
                }
            }

            double pivot = a[col, col];
            for (int c = 0; c < size; c++) { a[col, c] /= pivot; inv[col, c] /= pivot; }

            for (int r = 0; r < size; r++)
            {
                if (r == col) continue;
                double factor = a[r, col];
                if (factor == 0.0) continue;
                for (int c = 0; c < size; c++)
                {
                    a[r, c] -= factor * a[col, c];
                    inv[r, c] -= factor * inv[col, c];
                }
            }
        }
        return inv;
    }
}
