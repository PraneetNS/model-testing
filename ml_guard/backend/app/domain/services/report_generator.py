
from typing import Dict, Any, List
from datetime import datetime

class ReportGenerator:
    """
    Generates HTML reports from test results using Jinja2 templates (if minimal, or just f-strings).
    """
    
    def generate_html(self, run_id: str, results: List[Dict], output_path: str = "report.html"):
        """
        Generates a sleek, interactive HTML report using f-strings and basic HTML/Tailwind.
        """
        
        # Calculate summary stats
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "passed")
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        # Group by category
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        # Build category HTML
        category_sections = ""
        for cat, tests in categories.items():
            rows = ""
            for t in tests:
                # Color coding
                if t["status"] == "passed":
                    status_badge = '<span class="px-2 py-1 rounded-full text-xs font-bold bg-green-100 text-green-800">PASSED</span>'
                elif t["status"] == "warning":
                    status_badge = '<span class="px-2 py-1 rounded-full text-xs font-bold bg-yellow-100 text-yellow-800">WARNING</span>'
                else:
                    status_badge = '<span class="px-2 py-1 rounded-full text-xs font-bold bg-red-100 text-red-800">FAILED</span>'

                rows += f"""
                <tr class="border-b last:border-0 hover:bg-gray-50 transition-colors">
                    <td class="py-3 px-4 text-sm font-medium text-gray-900">{t["test_name"]}</td>
                    <td class="py-3 px-4">{status_badge}</td>
                    <td class="py-3 px-4 text-sm text-gray-600">{t["message"]}</td>
                    <td class="py-3 px-4 text-sm text-right font-mono text-gray-500">{t["score"]:.2f}</td>
                </tr>
                """
            
            category_sections += f"""
            <div class="mb-8 bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                <div class="bg-gray-50/50 px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                    <h3 class="text-lg font-semibold text-gray-800 capitalize flex items-center gap-2">
                        {cat.replace("_", " ")}
                        <span class="text-xs bg-gray-200 text-gray-600 px-2 py-0.5 rounded-full">{len(tests)}</span>
                    </h3>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="text-xs font-semibold text-gray-500 uppercase bg-gray-50/30 border-b border-gray-100">
                                <th class="py-3 px-4 w-1/3">Test Name</th>
                                <th class="py-3 px-4 w-24">Status</th>
                                <th class="py-3 px-4">Details</th>
                                <th class="py-3 px-4 w-20 text-right">Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        # Full HTML Template
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Guard Report - {run_id}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>body {{ font-family: 'Inter', sans-serif; }}</style>
</head>
<body class="bg-slate-50 text-slate-900 antialiased min-h-screen p-8">

    <div class="max-w-5xl mx-auto space-y-8">
        
        <!-- Header -->
        <header class="flex justify-between items-end border-b border-gray-200 pb-6">
            <div>
                <div class="flex items-center gap-3 mb-2">
                    <span class="text-2xl">🛡️</span>
                    <h1 class="text-3xl font-bold text-slate-900 tracking-tight">ML Guard Report</h1>
                </div>
                <p class="text-sm text-slate-500 font-medium">Run ID: <span class="font-mono text-slate-700">{run_id}</span></p>
                <p class="text-xs text-slate-400 mt-1">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
            </div>
            
            <div class="text-right">
                 <div class="flex items-baseline justify-end gap-1">
                    <span class="text-5xl font-extrabold {'text-green-600' if pass_rate >= 80 else 'text-red-500'}">
                        {pass_rate:.0f}%
                    </span>
                    <span class="text-sm font-semibold text-slate-400 uppercase tracking-wider">Pass Rate</span>
                </div>
            </div>
        </header>

        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="bg-white p-5 rounded-xl shadow-sm border border-gray-200 flex flex-col justify-between">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Total Tests</span>
                <span class="text-3xl font-bold text-slate-700 mt-2">{total}</span>
            </div>
            <div class="bg-white p-5 rounded-xl shadow-sm border border-gray-200 flex flex-col justify-between">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Passed</span>
                <span class="text-3xl font-bold text-green-600 mt-2">{passed}</span>
            </div>
            <div class="bg-white p-5 rounded-xl shadow-sm border border-gray-200 flex flex-col justify-between">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Failed</span>
                <span class="text-3xl font-bold text-red-600 mt-2">{failed}</span>
            </div>
            <div class="bg-white p-5 rounded-xl shadow-sm border border-gray-200 flex flex-col justify-between">
                 <span class="text-xs font-semibold text-slate-400 uppercase tracking-wide">Risk Level</span>
                 <span class="text-xl font-bold mt-2 {'text-green-600' if pass_rate >= 90 else 'text-yellow-600' if pass_rate >= 70 else 'text-red-600'}">
                    {'LOW' if pass_rate >= 90 else 'MEDIUM' if pass_rate >= 70 else 'HIGH'}
                 </span>
            </div>
        </div>

        <!-- Warning Area if critical failure -->
        {f'''
        <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg shadow-sm flex items-start gap-3">
            <span class="text-xl">⚠️</span>
            <div>
                <h3 class="font-bold text-red-800">Deployment Blocked</h3>
                <p class="text-sm text-red-700 mt-1">Critical tests failed. Please review the failures below before proceeding.</p>
            </div>
        </div>
        ''' if pass_rate < 70 else ''}

        <!-- Test Categories -->
        <div class="space-y-8">
            {category_sections}
        </div>

        <footer class="text-center text-xs text-slate-400 py-8 border-t border-gray-200 mt-12">
            Model Quality Gate • Power by Antigravity
        </footer>
    </div>
</body>
</html>
        """
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        return output_path
