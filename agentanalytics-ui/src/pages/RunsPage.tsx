import React, { useEffect, useMemo, useState } from "react";
import { Button, Card, Table, Tag, Typography } from "antd";
import { useNavigate } from "react-router-dom";
import { listRuns, RunSummary } from "../lib/api";

function fmtTime(ms?: number) {
  if (!ms) return "-";
  const d = new Date(ms);
  return d.toLocaleString();
}

export default function RunsPage() {
  const nav = useNavigate();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const r = await listRuns();
      r.sort((a, b) => (b.started_ms ?? 0) - (a.started_ms ?? 0));
      setRuns(r);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, []);

  const columns = useMemo(
    () => [
      {
        title: "Run ID",
        dataIndex: "run_id",
        render: (v: string) => (
          <Typography.Link onClick={() => nav(`/runs/${v}`)}>{v}</Typography.Link>
        )
      },
      {
        title: "Status",
        dataIndex: "status",
        sorter: (a: RunSummary, b: RunSummary) => String(a.status).localeCompare(String(b.status)),
        render: (v: string) => {
          const color = v === "FINISHED" ? "green" : v === "RUNNING" ? "blue" : v === "FAILED" ? "red" : "default";
          return <Tag color={color}>{v}</Tag>;
        }
      },
      {
        title: "Started",
        dataIndex: "started_ms",
        defaultSortOrder: "descend" as const,
        sorter: (a: RunSummary, b: RunSummary) => (a.started_ms ?? 0) - (b.started_ms ?? 0),
        render: (v: number) => fmtTime(v)
      },
      {
        title: "Finished",
        dataIndex: "finished_ms",
        sorter: (a: RunSummary, b: RunSummary) => (a.finished_ms ?? 0) - (b.finished_ms ?? 0),
        render: (v: number) => fmtTime(v)
      },
      {
        title: "Trace Count",
        dataIndex: "trace_count",
        sorter: (a: RunSummary, b: RunSummary) => (a.trace_count ?? 0) - (b.trace_count ?? 0),
        render: (v: number) => v ?? "-"
      },
      {
        title: "Plugins",
        dataIndex: "plugins",
        render: (v: string[]) => (v?.length ? v.map((p) => <Tag key={p}>{p}</Tag>) : "-")
      }
    ],
    [nav]
  );

  return (
    <Card
      title="Runs"
      extra={
        <div style={{ display: "flex", gap: 8 }}>
          <Button onClick={() => nav("/new")} type="primary">
            New Run
          </Button>
          <Button onClick={refresh}>Refresh</Button>
        </div>
      }
    >
      <Table
        rowKey="run_id"
        loading={loading}
        dataSource={runs}
        columns={columns as any}
        pagination={{ pageSize: 20 }}
      />
    </Card>
  );
}
