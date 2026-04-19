import React, { useMemo } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

// Palette — noise points (label -1) rendered in dark grey
const PALETTE = [
  '#6366f1', '#22d3ee', '#f59e0b', '#10b981',
  '#f43f5e', '#a78bfa', '#34d399', '#fb923c',
  '#38bdf8', '#e879f9',
]
const NOISE_COLOR = '#374151'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const { x, y, label } = payload[0].payload
  return (
    <div style={{
      background: '#1e2330', border: '1px solid #374151',
      borderRadius: 6, padding: '8px 12px', fontSize: 12
    }}>
      <p>x: {x.toFixed(3)}</p>
      <p>y: {y.toFixed(3)}</p>
      <p>cluster: {label === -1 ? 'noise' : label}</p>
    </div>
  )
}

export default function ClusterChart({ data, algorithm, nClusters, elapsedMs }) {
  const points = useMemo(() =>
    data.points.map((p, i) => ({ x: p.x, y: p.y, label: data.labels[i] })),
    [data]
  )

  // Group by label for separate Scatter series (needed for per-point colors)
  const grouped = useMemo(() => {
    const map = {}
    points.forEach(p => {
      if (!map[p.label]) map[p.label] = []
      map[p.label].push(p)
    })
    return map
  }, [points])

  return (
    <div style={{
      background: '#1a1d27', borderRadius: 12,
      padding: '16px 16px 8px', border: '1px solid #2d3348'
    }}>
      <div style={{ marginBottom: 10 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 }}>
          {algorithm.replace(/_/g, ' ')}
        </h3>
        <p style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
          {nClusters} cluster{nClusters !== 1 ? 's' : ''} &bull; {elapsedMs} ms
        </p>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart>
          <XAxis dataKey="x" hide />
          <YAxis dataKey="y" hide />
          <Tooltip content={<CustomTooltip />} />
          {Object.entries(grouped).map(([label, pts]) => (
            <Scatter
              key={label}
              data={pts}
              fill={label === '-1' ? NOISE_COLOR : PALETTE[Number(label) % PALETTE.length]}
              opacity={0.75}
              r={3}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
