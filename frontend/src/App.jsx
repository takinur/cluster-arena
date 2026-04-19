import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import ClusterChart from './components/ClusterChart'

const ALGO_LABELS = {
  kmeans: 'K-Means',
  affinity_propagation: 'Affinity Propagation',
  dbscan: 'DBSCAN',
  birch: 'BIRCH',
  optics: 'OPTICS',
  hdbscan: 'HDBSCAN',
  agglomerative: 'Agglomerative',
  spectral: 'Spectral',
}

export default function App() {
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState('all')
  const [error, setError] = useState(null)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.get('/cluster')
      setResults(data)
    } catch (e) {
      setError('API unreachable — make sure the FastAPI server is running on :8000')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  const visible = selected === 'all'
    ? results
    : results.filter(r => r.algorithm === selected)

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '32px 24px' }}>
      {/* Header */}
      <header style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, letterSpacing: -0.5 }}>
          Clustering Algorithm Benchmark
        </h1>
        <p style={{ color: '#64748b', marginTop: 6, fontSize: 14 }}>
          Comparing 8 unsupervised clustering algorithms on the same 2-D dataset
        </p>
      </header>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 28 }}>
        <button
          onClick={() => setSelected('all')}
          style={btnStyle(selected === 'all')}
        >All</button>
        {Object.entries(ALGO_LABELS).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            style={btnStyle(selected === key)}
          >{label}</button>
        ))}
        <button onClick={fetchAll} style={refreshBtnStyle}>
          {loading ? 'Running...' : 'Re-run'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: '#450a0a', border: '1px solid #991b1b',
          borderRadius: 8, padding: '12px 16px', marginBottom: 24, fontSize: 14
        }}>
          {error}
        </div>
      )}

      {/* Summary bar */}
      {results.length > 0 && (
        <div style={{
          display: 'flex', gap: 24, marginBottom: 28,
          overflowX: 'auto', paddingBottom: 4
        }}>
          {results.map(r => (
            <div key={r.algorithm} style={{
              background: '#1a1d27', borderRadius: 8, padding: '10px 16px',
              border: '1px solid #2d3348', minWidth: 130, flexShrink: 0,
              cursor: 'pointer', opacity: selected === r.algorithm || selected === 'all' ? 1 : 0.4,
              transition: 'opacity 0.2s',
            }} onClick={() => setSelected(r.algorithm === selected ? 'all' : r.algorithm)}>
              <p style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1 }}>
                {r.algorithm.replace(/_/g, ' ')}
              </p>
              <p style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{r.n_clusters}</p>
              <p style={{ fontSize: 11, color: '#64748b' }}>clusters &bull; {r.elapsed_ms} ms</p>
            </div>
          ))}
        </div>
      )}

      {/* Charts grid */}
      {loading && !results.length ? (
        <p style={{ color: '#64748b', textAlign: 'center', paddingTop: 80 }}>Running algorithms...</p>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
          gap: 20,
        }}>
          {visible.map(r => (
            <ClusterChart
              key={r.algorithm}
              data={r}
              algorithm={ALGO_LABELS[r.algorithm] ?? r.algorithm}
              nClusters={r.n_clusters}
              elapsedMs={r.elapsed_ms}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const btnStyle = (active) => ({
  padding: '7px 14px',
  borderRadius: 6,
  border: 'none',
  cursor: 'pointer',
  fontSize: 13,
  background: active ? '#6366f1' : '#1e2330',
  color: active ? '#fff' : '#94a3b8',
  transition: 'all 0.15s',
})

const refreshBtnStyle = {
  marginLeft: 'auto',
  padding: '7px 18px',
  borderRadius: 6,
  border: '1px solid #374151',
  cursor: 'pointer',
  fontSize: 13,
  background: 'transparent',
  color: '#94a3b8',
}
