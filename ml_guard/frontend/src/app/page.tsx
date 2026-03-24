import { Navbar } from '@/components/landing/Navbar';
import { Hero } from '@/components/landing/Hero';
import { Features } from '@/components/landing/Features';
import { HowItWorks } from '@/components/landing/HowItWorks';
import { Frameworks, DashboardPreview } from '@/components/landing/Frameworks';
import { Testimonials, Docs, Contact } from '@/components/landing/Closing';
import { Footer } from '@/components/landing/Footer';

export default function LandingPage() {
  return (
    <main className="bg-[#090A0C] min-h-screen">
      <Navbar />
      <Hero />
      <Frameworks />
      <Features />
      <DashboardPreview />
      <HowItWorks />
      <Testimonials />
      <Docs />
      <Contact />
      <Footer />

      {/* Global Cursor Glow Effect Implementation */}
      <div className="fixed inset-0 pointer-events-none z-[9999] opacity-20 bg-[radial-gradient(circle_400px_at_var(--mouse-x)_var(--mouse-y),rgba(249,115,22,0.15),transparent)]" />
    </main>
  );
}
