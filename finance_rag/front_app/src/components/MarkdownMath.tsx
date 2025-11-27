// src/components/MarkdownMath.tsx
'use client';

import React from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const INLINE_MATH_SAFE = /\$(?!\s*[+\-]?\d)([\s\S]*?)(?<!\\)\$/g;

/** Turn '\$' back into '$' *outside* math spans only. */
function fixCurrencyOutsideMath(md: string): string {
  if (!md) return md;
  const slots: string[] = [];

  // Mask $$...$$
  let masked = md.replace(/\$\$([\s\S]*?)\$\$/g, (_m, body) => {
    const id = slots.push(`$$${body}$$`) - 1;
    return `@@MB${id}@@`;
  });

  // Mask $...$
  masked = masked.replace(INLINE_MATH_SAFE, (_m, body) => {
    const id = slots.push(`$${body}$`) - 1;
    return `@@MI${id}@@`;
  });

  // Outside math only: unescape \$ → $
  masked = masked.replace(/\\+\$/g, '$');

  // Unmask
  masked = masked
    .replace(/@@MB(\d+)@@/g, (_m, i) => slots[Number(i)] ?? '')
    .replace(/@@MI(\d+)@@/g, (_m, i) => slots[Number(i)] ?? '');

  return masked;
}

/** Replace '\$' *inside* math with '\text{\$}' so KaTeX accepts it. */
function fixDollarInsideMath(md: string): string {
  if (!md) return md;
  const fix = (b: string) => String(b).replace(/\\\$/g, '\\text{\\$}');
  md = md.replace(/\$\$([\s\S]*?)\$\$/g, (_m, body) => `$$${fix(body)}$$`);
  md = md.replace(INLINE_MATH_SAFE, (_m, body) => `$${fix(body)}$`);
  return md;
}


/** Final light preprocess before Markdown → KaTeX render. */
function preprocess(md: string): string {
  let t = md || '';
  t = fixDollarInsideMath(t); 
  t = fixCurrencyOutsideMath(t); 
  return t;
}

const components: Components = {
  p: (props) => {
    const text = String(React.Children.toArray(props.children).join('')).trim();
    if (!text) return null; 
    return <p {...props} className="whitespace-normal break-words my-0" />;
  },
  li: (props) => <li {...props} className="my-0" />,
  code(props) {
    const { inline, className, children, ...rest } = props as any;
    return inline
      ? <code className={className} {...rest}>{children}</code>
      : (
          <pre className="overflow-x-auto my-2">
            <code className={className} {...rest}>{children}</code>
          </pre>
        );
  },
};

type Props = {
  text?: string;
  children?: string;
  className?: string;
};

export default function MarkdownMath({ text, children, className }: Props) {
  const content = preprocess(text ?? children ?? '');
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [rehypeKatex, { throwOnError: false, strict: 'ignore' as const }],
        ]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
