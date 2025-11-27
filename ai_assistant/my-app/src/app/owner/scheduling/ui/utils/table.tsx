// src/app/owner/scheduling/ui/utils/table.tsx
import React, { ReactNode } from "react";

export function Th({ children, className = "" }:{ children?: ReactNode; className?: string }) {
  return <th className={`text-left px-3 py-2 border-r last:border-r-0 ${className}`}>{children}</th>;
}
export function Td(
  { children, colSpan, className = "", ...rest }:
  React.PropsWithChildren<{ colSpan?: number; className?: string } & React.TdHTMLAttributes<HTMLTableCellElement>>
) {
  return (
    <td className={`px-3 py-2 border-r last:border-r-0 ${className}`} colSpan={colSpan} {...rest}>
      {children}
    </td>
  );
}
