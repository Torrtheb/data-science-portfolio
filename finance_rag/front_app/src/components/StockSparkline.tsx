import React, { useMemo } from 'react';

type Props = {
  times: number[];
  closes: number[];
  className?: string;
  showAxes?: boolean;
};

export default function StockSparkline({ times, closes, className, showAxes = false }: Props) {
  const pts = useMemo(() => {
    const n = Math.min(times?.length || 0, closes?.length || 0);
    if (n < 2) return null;

    const rows: Array<{ t: number; c: number }> = [];
    for (let i = 0; i < n; i++) {
      const t = times[i];
      const c = closes[i];
      if (typeof t === 'number' && isFinite(t) && typeof c === 'number' && isFinite(c)) {
        rows.push({ t, c });
      }
    }
    if (rows.length < 2) return null;

    const allT = rows.map(r => r.t);
    const msLike = Math.max(...allT) > 1e12;
    const ts = msLike ? rows.map(r => ({ ...r, t: r.t / 1000 })) : rows;

    const tVals = ts.map(r => r.t);
    const cVals = ts.map(r => r.c);
    const tMin = Math.min(...tVals);
    const tMax = Math.max(...tVals);
    const cMin = Math.min(...cVals);
    const cMax = Math.max(...cVals);

    const tRange = tMax - tMin || 1;
    const cRange = cMax - cMin || 1;

    const VB_W = 100;
    const VB_H = 100;
    const M_LEFT = showAxes ? 16 : 4;
    const M_BOTTOM = showAxes ? 16 : 4;
    const CH_W = VB_W - M_LEFT - 4;
    const CH_H = VB_H - M_BOTTOM - 4;

    const points = ts.map(({ t, c }) => {
      const x = M_LEFT + ((t - tMin) / tRange) * CH_W;
      const y = VB_H - M_BOTTOM - ((c - cMin) / cRange) * CH_H;
      return `${x},${y}`;
    });

    const flat = cMax === cMin;
    const flatY = VB_H - M_BOTTOM - CH_H / 2;

    const fmtDate = (sec: number) =>
      new Date(sec * 1000).toLocaleDateString(undefined, {
        month: 'short',
        day: tRange < 86400 * 120 ? 'numeric' : undefined,
        year: tRange > 86400 * 370 ? '2-digit' : undefined,
      });

    const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

    return {
      ptsStr: points.join(' '),
      VB_W, VB_H, M_LEFT, M_BOTTOM, CH_W, CH_H,
      flat, flatY,
      xLabels: showAxes
        ? [
            { x: M_LEFT, label: fmtDate(tMin), anchor: 'start' },
            { x: M_LEFT + CH_W / 2, label: fmtDate(tMin + tRange / 2), anchor: 'middle' },
            { x: M_LEFT + CH_W, label: fmtDate(tMax), anchor: 'end' },
          ]
        : [],
      yLabels: showAxes
        ? [
            { y: VB_H - M_BOTTOM, label: nf.format(cMin) },
            { y: VB_H - M_BOTTOM - CH_H / 2, label: nf.format(cMin + cRange / 2) },
            { y: VB_H - M_BOTTOM - CH_H, label: nf.format(cMax) },
          ]
        : [],
    };
  }, [times, closes, showAxes]);

  if (!pts) return <div className="text-xs text-gray-400">No chart</div>;

  const { ptsStr, VB_W, VB_H, M_LEFT, M_BOTTOM, CH_W, CH_H, flat, flatY, xLabels, yLabels } = pts;

  return (
    <div className={`relative ${className ?? ''}`}>
      <svg
        className="w-full h-full"
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        preserveAspectRatio="none"
        role="img"
      >
        <title>Price sparkline</title>

        {flat ? (
          <line
            x1={M_LEFT}
            x2={M_LEFT + CH_W}
            y1={flatY}
            y2={flatY}
            stroke="currentColor"
            strokeOpacity={0.95}
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            strokeLinecap="round"
          />
        ) : (
          <polyline
            fill="none"
            stroke="currentColor"
            strokeOpacity={0.95}
            strokeWidth={2}               
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={ptsStr}
            shapeRendering="geometricPrecision"
          />
        )}

        {/* baseline */}
        <line
          x1={M_LEFT}
          x2={M_LEFT + CH_W}
          y1={VB_H - M_BOTTOM}
          y2={VB_H - M_BOTTOM}
          stroke="currentColor"
          strokeOpacity={0.12}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      {/* Axes labels */}
      {showAxes && (
        <>
          {xLabels.map((t, i) => (
            <div
              key={`x-${i}`}
              className="absolute text-[10px] text-gray-500"
              style={{
                left: `${(t.x / VB_W) * 100}%`,
                bottom: 0,
                transform: `translateX(${t.anchor === 'middle' ? '-50%' : t.anchor === 'end' ? '-100%' : '0'})`,
              }}
            >
              {t.label}
            </div>
          ))}
          {yLabels.map((t, i) => (
            <div
              key={`y-${i}`}
              className="absolute text-[10px] text-gray-500"
              style={{
                left: 0,
                top: `${(t.y / VB_H) * 100}%`,
                transform: 'translateY(-50%)',
              }}
            >
              {t.label}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
