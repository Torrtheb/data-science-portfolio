require("dotenv").config({ path: "./.env.local" });
require("dotenv").config({ path: "./.env" });

const { PrismaClient } = require("@prisma/client");
const bcrypt = require("bcryptjs");
const p = new PrismaClient();

(async () => {
  const email = process.env.OWNER_EMAIL;
  const tz = process.env.OWNER_TIMEZONE || "America/Toronto";
  if (!email) throw new Error("OWNER_EMAIL not set in frontend/.env.local");

  // Optional: set an initial password for credentials login
  const plain = process.env.OWNER_PASSWORD;
  const setPassword = !!plain;
  const hash = plain ? bcrypt.hashSync(plain, 10) : undefined;

  let u = await p.user.findUnique({ where: { email } });
  if (!u) {
    u = await p.user.create({
      data: {
        email,
        name: "Owner",
        role: "OWNER",
        timezone: tz,
        ...(setPassword ? { password: hash, emailVerified: new Date() } : {}),
      },
    });
  } else {
    const update = { role: "OWNER", timezone: tz };
    if (setPassword) Object.assign(update, { password: hash, emailVerified: new Date() });
    u = await p.user.update({ where: { id: u.id }, data: update });
  }
  console.log("Owner:", { id: u.id, email: u.email, role: u.role, timezone: u.timezone });
  await p.$disconnect();
})().catch(async (e) => {
  console.error(e);
  process.exit(1);
});
