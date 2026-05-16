'use client';

import { cn } from '@/lib/utils';

interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (value: unknown, row: T) => React.ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  className?: string;
  emptyMessage?: string;
}

export function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  className,
  emptyMessage = 'No data available',
}: DataTableProps<T>) {
  return (
    <div className={cn('w-full overflow-x-auto', className)}>
      <table className="w-full border-collapse">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={cn(
                  'text-left pb-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted',
                  'border-b border-stone',
                  col.className
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="py-8 text-center text-sm text-muted"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row, i) => (
              <tr
                key={i}
                className={cn(
                  'border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors duration-100',
                  i % 2 === 0 ? 'bg-white' : 'bg-ivory'
                )}
              >
                {columns.map((col) => {
                  const val = row[col.key as keyof T];
                  return (
                    <td
                      key={String(col.key)}
                      className={cn(
                        'py-3 pr-4 text-[13px] text-ink-soft',
                        col.className
                      )}
                    >
                      {col.render ? col.render(val, row) : String(val ?? '')}
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default DataTable;
