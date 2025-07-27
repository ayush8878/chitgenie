const { onRequest } = require('firebase-functions/v2/https');
const { initializeApp } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const { BigQuery } = require('@google-cloud/bigquery');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize Firebase Admin
initializeApp();
const db = getFirestore();

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

// Initialize BigQuery
const bigquery = new BigQuery({
  projectId: process.env.GOOGLE_CLOUD_PROJECT,
});

class GeminiBigQueryBot {
  constructor(projectId, datasetId, tableId) {
    this.projectId = projectId;
    this.datasetId = datasetId;
    this.tableId = tableId;
    this.llm = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
    this.bigquery = bigquery;
  }

  async getTableSchema() {
    try {
      const [metadata] = await this.bigquery
        .dataset(this.datasetId)
        .table(this.tableId)
        .getMetadata();

      return metadata.schema.fields
        .map(field => `- ${field.name} (${field.type}): ${field.description || 'No description'}`)
        .join('\n');
    } catch (error) {
      console.error('Error getting table schema:', error);
      return `Could not fetch schema: ${error.message}`;
    }
  }

  async generateSql(naturalLanguageQuery) {
    const prompt = `You are an expert read only SQL generator for BigQuery. Convert the following natural language query to SQL.

Table Information:
- Table: \`${this.projectId}.${this.datasetId}.${this.tableId}\`
- Schema:
${await this.getTableSchema()}

Business Context:
- This is an invoices table with financial data
- "overdue" means status = 'overdue' or due_date < CURRENT_DATE() and status != 'paid'
- "pending" means status = 'pending'
- "paid" means status = 'paid'
- Use proper BigQuery functions like DATE_SUB, DATE_TRUNC for date operations
- Always limit results to prevent large queries (use LIMIT 100 unless specifically asked for more)

Natural Language Query: "${naturalLanguageQuery}"

Generate ONLY the SQL query, no explanation. Use proper BigQuery syntax.

SQL:
`;

    try {
      const result = await this.llm.generateContent(prompt);
      const response = await result.response;
      let sql = response.text().trim();
      
      // Clean up the response
      sql = sql.replace(/```sql\n?/, '').replace(/```\n?/, '').trim();
      
      console.log(`Generated SQL for query "${naturalLanguageQuery}": ${sql}`);
      return sql;
    } catch (error) {
      console.error('Error generating SQL:', error);
      throw new Error(`Failed to generate SQL: ${error.message}`);
    }
  }

  async executeQuery(sql) {
    try {
      if (!this.isSafeQuery(sql)) {
        throw new Error('Query contains potentially unsafe operations');
      }

      const options = {
        query: sql,
        useLegacySql: false,
        location: 'US',
        maxResults: 1000,
      };

      const [rows] = await this.bigquery.query(options);
      return rows;
    } catch (error) {
      console.error('Error executing query:', error);
      throw new Error(`Query execution failed: ${error.message}`);
    }
  }

  isSafeQuery(sql) {
    // Basic safety check - in production, implement more robust checks
    const unsafePatterns = [
      /(?:^|;)\s*(?:drop|delete|insert|update|create|alter|truncate|grant|revoke|deny|exec|execute|shutdown|use|backup|restore|load\s+data)/i,
      /\b(?:xp_cmdshell|sp_configure|xp_reg(?:read|write|delete(?:value|key))|sp_oacreate|sp_oamethod|sp_adduser|sp_addrole)\b/i,
      /\b(into\s+(outfile|dumpfile))\b/i,
    ];

    return !unsafePatterns.some(pattern => pattern.test(sql));
  }
}

// HTTP Cloud Function
exports.queryBigQuery = onRequest(async (req, res) => {
  // Set CORS headers
  res.set('Access-Control-Allow-Origin', '*');
  
  if (req.method === 'OPTIONS') {
    // Send response to OPTIONS requests
    res.set('Access-Control-Allow-Methods', 'POST');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.set('Access-Control-Max-Age', '3600');
    res.status(204).send('');
    return;
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).send('Method Not Allowed');
  }

  try {
    const { query, projectId, datasetId, tableId } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query parameter is required' });
    }

    const bot = new GeminiBigQueryBot(
      projectId || process.env.DEFAULT_PROJECT_ID,
      datasetId || process.env.DEFAULT_DATASET_ID,
      tableId || process.env.DEFAULT_TABLE_ID
    );

    const sql = await bot.generateSql(query);
    const results = await bot.executeQuery(sql);

    return res.status(200).json({
      query,
      sql,
      results,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error processing request:', error);
    return res.status(500).json({
      error: error.message || 'Internal Server Error',
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});
