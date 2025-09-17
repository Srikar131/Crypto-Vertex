// frontend/app/api/cryptonews/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const symbol = searchParams.get("symbol") || "btc";
  const url = `https://cryptopanic.com/api/v1/posts/?auth_token=demo&currencies=${symbol}&public=true`;

  try {
    const res = await fetch(url, { next: { revalidate: 60 } });
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json({ error: "Failed to fetch news." }, { status: 500 });
  }
}
